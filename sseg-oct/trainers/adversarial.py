from trainers.trainer import Trainer
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, Callable, Iterator, Tuple
from torch.utils.data import DataLoader
from utils.metrics import Metric
import os, pickle
from models.discriminator import Discriminator
from utils.constants import MAX_THREADS
import numpy as np


class AdversarialTrainer(Trainer):

    G_SUFFIX = '_generator'
    D_SUFFIX = '_discriminator'

    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            device: torch.device,
            optimizer: Optimizer,
            criterion,
            metrics: Dict[str, Callable],
            scheduler=None,
    ):
        """
        Initialization of the Trainer.
        :param model: Pytorch model to train.
        :param device: Device of the model.
        :param optimizer: Pytorch optimizer.
        :param criterion: Loss function.
        :param scheduler: [Optional] Learning rate scheduler.
        """
        self.generator = generator
        self.discriminator = discriminator.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.scheduler = scheduler
        self.verbose = False
        self.threshold = None

        self.discriminator_loss = nn.CrossEntropyLoss()

    def forward_step(self, loader: DataLoader, optimize: bool):
        global_loss = 0
        metric = Metric(self.metrics)

        for i, batch in enumerate(loader):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            # forward pass for generator
            outputs1 = self.generator(inputs)
            gloss = self.criterion(outputs1.flatten(), targets.flatten())

            # compute best_threshold
            if optimize:
                self.update_metric(outputs1.flatten(), targets.flatten(), metric)
                outputs1 = (outputs1 > self.threshold) * 1.0
            else:
                self.compute_metric(outputs1, targets, metric)
                outputs1 = (outputs1 > self.threshold) * 1.0

            # forward pass for discriminator
            outputs2 = self.discriminator(torch.concat([targets, outputs1], dim=0).to(self.device))
            dtargets = torch.concat([torch.ones(targets.shape[0]), torch.zeros(outputs1.shape[0])]).to(self.device)

            outputs2, dtargets = map(lambda x: x.to(torch.float32).flatten(), [outputs2, dtargets])
            dloss = self.discriminator_loss(outputs2, dtargets)

            loss = gloss + dloss
            global_loss += loss.item()

            if optimize:
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return global_loss / len(loader), metric


    def load_checkpoint(self, save_model):
        g_checkpoint = torch.load(f'{save_model}{AdversarialTrainer.G_SUFFIX}.pt')
        d_checkpoint = torch.load(f'{save_model}{AdversarialTrainer.D_SUFFIX}.pt')
        self.generator.load_state_dict(g_checkpoint['model_state_dict'])
        self.discriminator.load_state_dict(d_checkpoint['model_state_dict'])


    def save_checkpoint(self, epoch: int, loss: torch.Tensor, save_model: str):
        torch.save(
            {'epoch': epoch, 'model_state_dict': self.generator.state_dict(),
             'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss,
             'threshold': self.threshold},
            save_model.removesuffix('.pt') + AdversarialTrainer.G_SUFFIX + '.pt'
        )
        torch.save(
            {'epoch': epoch, 'model_state_dict': self.discriminator.state_dict(),
             'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss},
            save_model.removesuffix('.pt') + AdversarialTrainer.D_SUFFIX + '.pt'
        )

    @classmethod
    def kfold(
            self,
            generator_builder: Callable, image_size: Tuple[int], device: torch.device, optimizer_builder: Callable,
            criterion, metrics: Dict[str, Callable],
            kfold: Iterator, batch_size: int,
            num_epochs: int,
            patience: int,
            checkpoint_metric: str,
            save_model: str,
            verbose: bool
          ):
        fold_results = list()

        for fold, (train, val) in enumerate(iter(kfold)):
            if verbose:
                print(f'Training for fold {fold+1}')

            generator = generator_builder().to(device)
            discriminator = Discriminator(in_channels=1, img_size=image_size).to(device)
            optimizer = optimizer_builder(list(generator.parameters()) + list(discriminator.parameters()))
            trainer = AdversarialTrainer(generator, discriminator, device, optimizer, criterion, metrics)

            trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=MAX_THREADS)
            valloader = DataLoader(val, batch_size=batch_size, num_workers=MAX_THREADS)

            trainer.train(trainloader, valloader, num_epochs=num_epochs, patience=patience,
                       save_model=f'{save_model}/fold{fold}', checkpoint_metric=checkpoint_metric, verbose=verbose)
            val_loss, val_metrics = trainer.validate(valloader)
            if fold != 0:
                os.remove(f'{save_model}/fold{fold}_discriminator.pt')
                os.remove(f'{save_model}/fold{fold}_generator.pt')
            fold_results.append((val_loss, val_metrics))
            print('-'*80)
        with open(f'{save_model}/fold-results.pickle', 'wb') as file:
            pickle.dump(obj=fold_results, file=file)

        return fold_results


