from trainers.trainer import Trainer
from torch.utils.data import DataLoader
from utils.metrics import Metric
import torch, os, pickle
from typing import Callable, Iterator, Dict
from utils.constants import MAX_THREADS


class ConvolutionalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward_step(self, loader: DataLoader, optimize: bool):
        global_loss = 0
        metric = Metric(self.metrics)

        for i, batch in enumerate(loader):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device).flatten()

            # forward pass
            outputs = self.model(inputs).flatten()
            loss = self.criterion(outputs, targets)

            global_loss += loss.item()

            if optimize:
                # backward pass
                self.update_metric(outputs, targets, metric)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                self.compute_metric(outputs, targets, metric)

        if self.scheduler:
            self.scheduler.step()

        return global_loss/len(loader), metric

    @classmethod
    def kfold(
            self,
            model_builder: Callable, device: torch.device, optimizer_builder: Callable,
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

            model = model_builder().to(device)
            optimizer = optimizer_builder(model.parameters())
            trainer = ConvolutionalTrainer(model, device, optimizer, criterion, metrics)

            trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=MAX_THREADS)
            valloader = DataLoader(val, batch_size=batch_size, num_workers=MAX_THREADS)

            trainer.train(trainloader, valloader,
                          num_epochs=num_epochs, patience=patience,
                          save_model=f'{save_model}/fold{fold}.pt',
                          checkpoint_metric=checkpoint_metric, verbose=verbose)
            val_loss, val_metrics = trainer.validate(valloader)
            if fold != 0:
                os.remove(f'{save_model}/fold{fold}.pt')
            fold_results.append((val_loss, val_metrics))
            print('-'*80)
        with open(f'{save_model}/fold-results.pickle', 'wb') as file:
            pickle.dump(obj=fold_results, file=file)

        return fold_results
