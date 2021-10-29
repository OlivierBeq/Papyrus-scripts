# -*- coding: utf-8 -*-

import os
import time
from itertools import cycle
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, IterableDataset as PandasIterableDataset


def cuda(var: nn.Module):
    """Move model parameters and buffers to GPU if a GPU is available.

    Originates from Xuhan Liu's DrugEx version 1 (https://github.com/XuhanLiu/DrugEx/tree/1.0)

    :param var: torch.nn.Module derived class to be trained on GPU (or CPU if not GPU available)
    """
    if T.cuda.is_available():
        return var.cuda()
    return var


def Variable(tensor: Union[T.Tensor, np.ndarray, List]):
    """Transform a list or numpy array into a pytorch tensor on GPU (if available).

    Originates from Xuhan Liu's DrugEx version 1 (https://github.com/XuhanLiu/DrugEx/tree/1.0)
    Original documentation: Wrapper for torch.autograd.Variable that also accepts
                            numpy arrays directly and automatically assigns it to
                            the GPU. Be aware in some cases operations are better
                            left to the CPU.
    :param tensor: the list, numpy array or pytorch tensor to be sent to GPU (if available)
    """
    if isinstance(tensor, np.ndarray):
        tensor = T.from_numpy(tensor)
    if isinstance(tensor, list):
        tensor = T.Tensor(tensor)
    return cuda(T.autograd.Variable(tensor))


class BaseNN(nn.Module):
    def __init__(self, out: str, epochs:int=100, lr:float=1e-3, early_stop: int = 100, batch_size: int =1024):
        """Base class for neural networks.

        Architecture is derived from https://doi.org/10.1186/s13321-017-0232-0

        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        """
        if not os.path.isdir(out):
            raise ValueError('output directory does not exist')
        super().__init__()
        self.out = out
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stop = early_stop

    def set_validation(self, X: Union[Iterator, pd.DataFrame], y : Union[Iterator, pd.Series]):
        """Set the validation set to be used during fitting.

        :param X: features to predict y from
        :param y: feature to be predicted (dependent variable)
        """
        if type(X) != type(y):
            raise ValueError('X and y must have the same type (i.e. either Iterator or pandas dataframe)')
        # Get data loaders
        if isinstance(X, pd.DataFrame):
            self.loader_valid = loader_from_dataframe(X, y, batch_size=self.batch_size)
        else:
            self.loader_valid = loader_from_iterator(X, y, batch_size=self.batch_size)

    def fit(self, X: Union[Iterator, pd.DataFrame], y : Union[Iterator, pd.Series]):
        """Fit neural network with training set and optimize for loss on validation set.

        :param X: features to predict y from
        :param y: feature to be predicted (dependent variable)
        """
        if type(X) != type(y):
            raise ValueError('X and y must have the same type (i.e. either Iterator or pandas dataframe)')
        # Get data loaders
        if isinstance(X, pd.DataFrame):
            loader_train = loader_from_dataframe(X, y, batch_size=self.batch_size)
        else:
            loader_train = loader_from_iterator(X, y, batch_size=self.batch_size)
        # Set optimizer
        if 'optim' in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        best_loss = np.inf
        last_save = 0
        # Set up output folder
        if not (os.path.exists(self.out) and os.path.isdir(self.out)):
            os.mkdir(self.out)
        # Log file
        log = open(os.path.join(self.out, 'training.log', 'w'))
        for epoch in range(self.epochs):
            t0 = time.time()
            # Change learning rate according to epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr * (1 - 1 / self.epochs) ** (epoch * 10)
            # Train epoch over all batches
            for i, (Xb, yb) in enumerate(loader_train):
                Xb, yb = Variable(Xb), Variable(yb)
                optimizer.zero_grad()
                y_ = self.forward(Xb, istrain=True)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = self.criterion(y_, yb)
                loss.backward()
                optimizer.step()
            # Calculate loss and log
            loss_valid = self.evaluate(self.loader_valid)
            print(f'[Epoch: {epoch}/{self.epochs}] {time.time() - t0:.1f}s '
                  f'loss_train: {loss.data[0]:f} loss_valid: {loss_valid:f}', file=log)
            if loss_valid < best_loss:
                T.save(self.state_dict(), os.path.join(self.out, 'model.pkg'))
                print(f'[Performance] loss_valid improved from {best_loss:f} to {loss_valid:f}, '
                      'Saved model to model.pkg', file=log)
                best_loss = loss_valid
                last_save = epoch
            else:
                print('[Performance] loss_valid did not improved.', file=log)
                # Early stop if no improvement for some time
                if epoch - last_save > self.early_stop:
                    break
        log.close()
        self.load_state_dict(T.load(self.out + '.pkg'))

    def evaluate(self, loader):
        """Calculate loss according to criterion function

        :param loader: data loader of the validation set
        """
        loss = 0
        for Xb, yb in loader:
            Xb, yb = Variable(Xb), Variable(yb)
            y_ = self.forward(Xb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += self.criterion(y_, yb).data[0]
        return loss / len(loader)

    def predict(self, loader):
        """Predict outcome for the incoming data

        :param loader: data loader of the data to make predictions from
        """
        score = []
        for Xb, yb in loader:
            Xb = Variable(Xb)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return T.cat(score, dim=0).numpy()


class SingleTaskNNClassifier(BaseNN):
    def __init__(self, n_dim: int, n_class: int, out: str, epochs:int=100, lr:float=1e-3, early_stop: int = 100, batch_size: int =1024):
        """Neural Network classifier to predict a unique endpoint.

        Architecture is derived from https://doi.org/10.1186/s13321-017-0232-0

        :param n_dim: number of input parameters
        :param n_class: number of classes (e.g. 2 for binary endpoint)
        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        """
        if n_class < 2:
            raise ValueError('can only perform binary or multi-classes predictions')
        super(SingleTaskNNClassifier, self).__init__(out, epochs, lr, early_stop, batch_size)
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, n_class)
        if n_class == 2:
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax()
        cuda(self)

    def forward(self, X, istrain=False):
        """Calculate model output from input data.

        :param X: input data
        :param istrain: whether called during training, to activate dropout
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        y = self.activation(self.fc3(y))
        return y


class SingleTaskNNRegressor(BaseNN):
    def __init__(self, n_dim:int, out: str, epochs:int=100, lr:float=1e-3, early_stop: int = 100, batch_size: int =1024):
        """Neural Network regressor to predict a unique endpoint.

        Architecture is adapted from https://doi.org/10.1186/s13321-017-0232-0 for regression

        :param n_dim: number of input parameters
        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        """
        super(SingleTaskNNRegressor, self).__init__(out, epochs, lr, early_stop, batch_size)
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, 1)
        self.criterion = nn.MSELoss()
        cuda(self)

    def forward(self, X, istrain=False):
        """Calculate model output from input data.

        :param X: input data
        :param istrain: whether called during training, to activate dropout
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        y = self.fc3(y)
        return y


class MultiTaskNNClassifier(BaseNN):
    def __init__(self, n_dim:int, n_task:int, out: str, epochs:int=100, lr:float=1e-3, early_stop: int = 100, batch_size: int =1024):
        """Neural Network classifier to predict multiple endpoints.

        Architecture is derived from https://doi.org/10.1186/s13321-017-0232-0

        :param n_dim: number of input parameters
        :param n_task: number of tasks to be predicted at the same time
        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        """
        if n_task < 2:
            raise ValueError('use SingleTaskNNClassifier for a single task')
        super(MultiTaskNNClassifier, self).__init__(out, epochs, lr, early_stop, batch_size)
        self.n_task = n_task
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.output = nn.Linear(2000, n_task)
        self.criterion = nn.BCELoss()
        self.activation = nn.Sigmoid()
        cuda(self)

    def forward(self, X, istrain=False):
        """Calculate model output from input data.

        :param X: input data
        :param istrain: whether called during training, to activate dropout
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        y = self.activation(self.output(y))
        return y


class MultiTaskNNRegressor(BaseNN):
    def __init__(self, n_dim:int, n_task:int, out: str, epochs:int=100, lr:float=1e-3, early_stop: int = 100, batch_size: int =1024):
        """Neural Network regressor to predict multiple endpoints.

        Architecture is adapted from https://doi.org/10.1186/s13321-017-0232-0 for multi-task regression

        :param n_dim: number of input parameters
        :param n_task: number of tasks to be predicted at the same time
        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        """
        if n_task < 2:
            raise ValueError('use SingleTaskNNRegressor for a single task')
        super(MultiTaskNNRegressor, self).__init__(out, epochs, lr, early_stop, batch_size)
        self.n_task = n_task
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.bn0 = nn.BatchNorm1d(8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.output = nn.Linear(4000, n_task)
        self.criterion = nn.MSELoss()
        cuda(self)

    def forward(self, X, istrain=False):
        """Calculate model output from input data.

        :param X: input data
        :param istrain: whether called during training, to activate dropout
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        # y = F.sigmoid(self.fc2(y))
        # if istrain: y = self.dropout(y)
        y = self.output(y)
        return y


def loader_from_dataframe(X: pd.DataFrame, Y: Optional[pd.Series] = None, y_col: Optional[str] = None,
                          batch_size: int = 1024):
    """Get PyTorch dataloaders from pandas dataframes

    :param X: features to predict Y from
    :param Y: feature to be predicted (dependent variable)
    :param y_col: name of the column in X containing the depedent variable to be predicted
    :param batch_size: batch size of the data loader
    """
    if Y is None and y_col is None:
        raise ValueError('either Y or y_col must be specified')
    if Y is None:
        Y = X.pop(y_col)
    dataset = TensorDataset(T.Tensor(X), T.Tensor(Y))
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


def loader_from_iterator(X: Iterator, Y: Optional[pd.Series] = None, y_col: Optional[str] = None,
                         batch_size: int = 1024):
    """Get PyTorch dataloaders from iterators

    :param X: features to predict Y from
    :param Y: feature to be predicted (dependent variable)
    :param y_col: name of the column in X containing the depedent variable to be predicted
    :param batch_size: batch size of the data loader
    """
    if Y is None and y_col is None:
        raise ValueError('either Y or y_col must be specified')
    x_dataset = IterableDataset(X)
    y_dataset = IterableDataset(Y)
    return (DataLoader(x_dataset, batch_size=batch_size),
            DataLoader(y_dataset, batch_size=batch_size))


class IterableDataset(PandasIterableDataset):
    def __init__(self, iterator: Iterator):
        self.iterator = iterator

    def cycle(self):
        return cycle(self.iterator)

    def __iter__(self):
        return self.cycle()
