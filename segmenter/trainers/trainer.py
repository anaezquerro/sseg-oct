from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn as nn
import os
from utils.metrics import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Callable, Tuple
import numpy as np

MAX_THREADS = os.cpu_count()

class Trainer:
    """
    Implementation of a trainer to handle models training and validation.
    """

    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            optimizer: Optimizer,
            criterion,
            metrics: Dict[str, Callable],
            scheduler = None,
        ):
        """
        Initialization of the Trainer.
        :param model: Pytorch model to train.
        :param device: Device of the model.
        :param optimizer: Pytorch optimizer.
        :param criterion: Loss function.
        :param scheduler: [Optional] Learning rate scheduler.
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.scheduler = scheduler
        self.verbose = False
        self.threshold = None

    def forward_step(self, loader: DataLoader, optimizer: bool):
        raise NotImplementedError

    def update_metric(self, outputs: torch.Tensor, targets: torch.Tensor, metric: Metric):
        """
        Updates Metric object given the outputs and targets of the model and returns the optimal threshold. In order
        to test a grid of threshold, a linspace of 20 values is taken between the range of the output.
        :param outputs: Outputs of the model.
        :param targets: Targets.
        :param metric: Metric object to update metrics.
        """
        self.optimize_threshold(outputs, targets, thresholds=np.linspace(outputs.min().item(), outputs.max().item(), 20))
        metric((outputs > self.threshold)*1.0, targets)

    def optimize_threshold(self, outputs: torch.LongTensor, targets: torch.LongTensor, thresholds: torch.Tensor):
        """
        Computes the optimal threshold using IoU and a grid of possible values.
        """
        self.threshold = 0
        best_value = -1
        for threshold in thresholds:
            value = iou((outputs > threshold)*1.0, targets)
            if value > best_value:
                self.threshold = threshold
                best_value = value

    def compute_metric(self, outputs: torch.Tensor, targets: torch.Tensor, metric: Metric):
        """
        Updates metric with a given threshold.
        """
        metric((outputs > self.threshold) * 1.0, targets)

    def train(
            self,
            trainloader: DataLoader,
            valloader: DataLoader,
            num_epochs: int,
            patience: int,
            save_model: str,
            checkpoint_metric: str,
            verbose: bool = True,
            display: bool = False,
            save_plot: str = None
    ):
        """
        Model training.
        :param trainloader: DataLoader of the training set.
        :param valloader: DataLoader of the validation set.
        :param num_epochs: Number of training epochs.
        :param save_model: Path where model will be stored.
        :param verbose: Display or not the metrics at each epoch.
        :param display: Display the final plot of the training evolution.
        :param save_plot: Path where plots will be stored. If not specified (or None), it will not be stored.
        :returns: Last threshold
        """
        best_metric = -np.Inf
        noimprove = 0
        train_history = list()
        val_history = list()
        self.verbose = verbose

        for epoch in range(num_epochs):
            train_loss, train_metric = self.forward_step(trainloader, optimize=True)

            with torch.no_grad():
                val_loss, val_metric = self.forward_step(valloader, optimize=False)
                if verbose:
                    print(f'Epoch {epoch + 1}/{num_epochs}:\ntrain_loss={round(train_loss, 3)}, {repr(train_metric)}\nval_loss={round(val_loss, 3)}, {repr(val_metric)}')

                if best_metric < val_metric[checkpoint_metric]:
                    self.save_checkpoint(epoch, train_loss, save_model)
                    best_metric = val_metric[checkpoint_metric]
                else:
                    noimprove += 1
                    if noimprove == patience:
                        break
            train_history.append((train_loss, train_metric))
            val_history.append((val_loss, val_metric))

        self.load_checkpoint(save_model)

        if display:
            self.plot_history(train_history, val_history, save_plot)


    def load_checkpoint(self, save_model):
        """
        Loads a checkpoint inside Trainer by updating model weights and loading the optimal threshold.
        """
        checkpoint = torch.load(save_model, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']

    def save_checkpoint(self, epoch: int, loss: torch.Tensor, save_model: str):
        """
        Saves checkpoint model and threshold.
        """
        torch.save(
            {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
             'optimizer_state_dict': self.optimizer.state_dict(),
             'loss': loss, 'threshold': self.threshold,
             },
            save_model
        )

    def validate(self, valloader: DataLoader):
        with torch.no_grad():
            val_loss, val_metrics = self.forward_step(valloader, optimize=False)
        return val_loss, val_metrics


    def plot_history(self, train_metrics: List[Tuple[float, Metric]], val_metrics: List[Tuple[float, Metric]], save_plot: str = None):
        metrics = list(self.metrics.keys())
        train_history = zip(*map(lambda x: [x[0]] + [x[1][m] for m in metrics], train_metrics))
        val_history = zip(*map(lambda x: [x[0]] + [x[1][m] for m in metrics], val_metrics))

        metrics = ['loss'] + metrics
        train_history, val_history = map(lambda x: dict(zip(metrics, x)), [train_history, val_history])
        epochs = list(range(len(train_metrics)))

        colors = px.colors.qualitative.Plotly
        fig = make_subplots(rows=1, cols=2)
        for i, metric in enumerate(metrics):
            if metric == 'loss':
                fig.add_trace(go.Scatter(x=epochs, y=train_history[metric], mode='lines+markers',
                                         line=dict(dash='solid', color=colors[i]), name='train-loss'), row=1, col=2)
                fig.add_trace(go.Scatter(x=epochs, y=val_history[metric], mode='lines+markers',
                                         line=dict(dash='dot', color=colors[i]), name='val-loss'), row=1, col=2)
            else:
                fig.add_trace(go.Scatter(x=epochs, y=train_history[metric], mode='lines+markers',
                                         line=dict(dash='solid', color=colors[i]), name=metric,
                                         legendgroup='train', legendgrouptitle_text='train metrics'), row=1, col=1)
                fig.add_trace(go.Scatter(x=epochs, y=val_history[metric], mode='lines+markers',
                                         line=dict(dash='dot', color=colors[i]), name=metric, legendgroup='val',
                                         legendgrouptitle_text='val metrics'), row=1, col=1)

        fig.update_layout(height=600, width=1600, title='Training and validation history', legend=dict(groupclick="toggleitem"))
        fig.update_xaxes(title_text='epochs', row=1, col=1)
        fig.update_xaxes(title_text='epochs', row=1, col=2)
        fig.update_yaxes(title_text='performance', row=1, col=1)
        fig.update_yaxes(title_text='loss', row=1, col=2)
        fig.show()

        if save_plot:
            if save_plot.endswith('.pdf'):
                fig.write_image(save_plot)
            elif save_plot.endswith('.html'):
                fig.write_html(save_plot)
            else:
                raise NotImplementedError














