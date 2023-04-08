# -*- coding: utf-8 -*-

import os
import time
import random
import itertools
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.io.parsers import TextFileReader as PandasTextFileReader

try:
    import torch as T
    from torch import nn, optim
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset, IterableDataset as PandasIterableDataset
except ImportError as e:
    T = e
    nn = e
    # Placeholders
    T.Tensor = int
    nn.Module = list
    PandasIterableDataset = int


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


def set_seed(seed: Optional[int] = None) -> Optional[np.random.Generator]:
    """Set the internal seed of rnadom number generators for reproducibility."""
    if seed is None:
        return
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    T.cuda.manual_seed(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    return rng


class BaseNN(nn.Module):
    def __init__(self, out: str, epochs: int = 100, lr: float = 1e-3,
                 early_stop: int = 100, batch_size: int = 1024, dropout: float = 0.25,
                 random_seed: Optional[int] = None):
        """Base class for neural networks.

        Architecture is derived from https://doi.org/10.1186/s13321-017-0232-0

        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        :param dropout: fraction of randomly disabled neurons at each epoch during training
        :param random_seed: seed of random number generators
        """
        if isinstance(T, ImportError):
            raise ImportError('Some required dependencies are missing:\n\tpytorch')
        if not os.path.isdir(out):
            os.makedirs(out, exist_ok=True)
        super().__init__()
        self.fcl = nn.ModuleList()  # fully connected layers
        self.out = out
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stop = early_stop
        self.dropout = dropout
        self.rng = set_seed(random_seed)

    def set_validation(self, X: Union[Iterator, pd.DataFrame], y: Union[Iterator, pd.Series]):
        """Set the validation set to be used during fitting.

        :param X: features to predict y from
        :param y: feature to be predicted (dependent variable)
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)) and type(X) != type(y):
            raise ValueError('X and y must have the same type (i.e. either Iterator or pandas dataframe)')
        # Get data loaders
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            self.loader_valid = loader_from_dataframe(X, y, batch_size=self.batch_size)
        else:
            self.loader_valid = loader_from_iterator(X, y, batch_size=self.batch_size)

    def set_architecture(self, dimensions: List[int]):
        """Define the size of each fully connected linear hidden layer

        :param dimensions: dimensions of the layers
        """
        for i in range(len(dimensions) - 1):
            self.fcl.append(nn.Linear(dimensions[i], dimensions[i + 1]))
        T.save(self.state_dict(), os.path.join(self.out, 'empty_model.pkg'))

    def reset(self):
        """Reset weights and reload the initial state of the model"""
        self.load_state_dict(T.load(os.path.join(self.out, 'empty_model.pkg')))

    def fit(self, X: Union[Iterator, pd.DataFrame], y: Union[Iterator, pd.Series]):
        """Fit neural network with training set and optimize for loss on validation set.

        :param X: features to predict y from
        :param y: feature to be predicted (dependent variable)
        """
        if not self.fcl:
            raise ValueError('set architecture before fitting')
        if not isinstance(X, (pd.DataFrame, np.ndarray)) and type(X) != type(y):
            raise ValueError('X and y must have the same type (i.e. either Iterator or pandas dataframe)')
        # Set number of classes
        self.classes_ = sorted(set(y))
        # Get data loaders
        if isinstance(X, (pd.DataFrame, np.ndarray)):
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
        log = open(os.path.join(self.out, 'training_log.txt'), 'w')
        for epoch in range(self.epochs):
            t0 = time.perf_counter()
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
            print(f'[Epoch: {epoch + 1}/{self.epochs}] {time.perf_counter() - t0:.1f}s '
                  f'loss_train: {loss.item():f} loss_valid: {loss_valid:f}', file=log, flush=True)
            if loss_valid < best_loss:
                T.save(self.state_dict(), os.path.join(self.out, 'model.pkg'))
                print(f'[Performance] loss_valid improved from {best_loss:f} to {loss_valid:f}, '
                      'Saved model to model.pkg', file=log, flush=True)
                best_loss = loss_valid
                last_save = epoch
            else:
                print('[Performance] loss_valid did not improve.', file=log, flush=True)
                # Early stop if no improvement for some time
                if epoch - last_save > self.early_stop:
                    break
        log.close()
        self.load_state_dict(T.load(os.path.join(self.out, 'model.pkg')))

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
            loss += self.criterion(y_, yb).item()
        return loss / len(loader)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        """Predict outcome for the incoming data

        :param X: features to predict the endpoint(s) from
        """
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('X must be either a numpy array or a pandas dataframe')
        if isinstance(X, pd.DataFrame):
            y = X.iloc[:, 0]
        else:
            y = X[:, 0]
        loader = loader_from_dataframe(X, y, self.batch_size)
        score = []
        for Xb, _ in loader:
            Xb = Variable(Xb)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return T.cat(score, dim=0).numpy()


class SingleTaskNNClassifier(BaseNN):
    def __init__(self, out: str, epochs: int = 100, lr: float = 1e-3,
                 early_stop: int = 100, batch_size: int = 1024, dropout: float = 0.25,
                 random_seed: Optional[int] = None):
        """Neural Network classifier to predict a unique endpoint.

        Architecture is derived from https://doi.org/10.1186/s13321-017-0232-0

        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        :param dropout: fraction of randomly disabled neurons at each epoch during training
        :param random_seed: seed of random number generators
        """
        super(SingleTaskNNClassifier, self).__init__(out, epochs, lr, early_stop, batch_size, dropout, random_seed)
        self.dropoutl = nn.Dropout(self.dropout)
        # Consider binary classification as default
        self.criterion = nn.BCELoss()
        self.activation = nn.Sigmoid()

    def set_architecture(self, n_dim: int, n_class: int):
        """Set dimension of input and number of classes to be predicted.

        :param n_dim: number of input parameters
        :param n_class: number of one-hot encoded classes (i.e. 1 for binary endpoint not one-hot encoded)
        """
        if n_class < 1:
            raise ValueError('can only perform binary (n_class=1 or n_class=2)'
                             ' or multi-classes predictions (n_class>2)')
        super().set_architecture([n_dim, 8000, 4000, 2000, n_class])
        self._n_classes_ = n_class
        self._n_features_in_ = n_dim
        if n_class == 1:
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
        input = X
        for layer in self.fcl[:-1]:
            input = F.relu(layer(input))
            if istrain:
                input = self.dropoutl(input)
        return self.activation(self.fcl[-1](input))

    def predict_proba(self, X):
        """Predict class probabilities for the incoming data

        :param X: features to predict the endpoint probabilities from
        """
        y = super().predict(X)
        return y

    def predict(self, X):
        """Predict classes for the incoming data

        :param X: features to predict the endpoint(s) from
        """
        probas = self.predict_proba(X)
        return np.round(probas)


class SingleTaskNNRegressor(BaseNN):
    def __init__(self, out: str, epochs: int = 100, lr: float = 1e-3,
                 early_stop: int = 100, batch_size: int = 1024, dropout: float = 0.25,
                 random_seed: Optional[int] = None):
        """Neural Network regressor to predict a unique endpoint.

        Architecture is adapted from https://doi.org/10.1186/s13321-017-0232-0 for regression

        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        :param dropout: fraction of randomly disabled neurons at each epoch during training
        :param random_seed: seed of random number generators
        """
        super(SingleTaskNNRegressor, self).__init__(out, epochs, lr, early_stop, batch_size, dropout, random_seed)
        self.dropoutl = nn.Dropout(self.dropout)
        self.criterion = nn.MSELoss()

    def set_architecture(self, n_dim: int):
        """Set dimension of input.

        :param n_dim: number of input parameters
        """
        super().set_architecture([n_dim, 8000, 4000, 2000, 1])
        cuda(self)

    def forward(self, X, istrain=False):
        """Calculate model output from input data.

        :param X: input data
        :param istrain: whether called during training, to activate dropout
        """
        input = X
        for layer in self.fcl[:-1]:
            input = F.relu(layer(input))
            if istrain:
                input = self.dropoutl(input)
        return self.fcl[-1](input)


class MultiTaskNNClassifier(BaseNN):
    def __init__(self, out: str, epochs: int = 100, lr: float = 1e-3,
                 early_stop: int = 100, batch_size: int = 1024, dropout: float = 0.25,
                 random_seed: Optional[int] = None):
        """Neural Network classifier to predict multiple endpoints.

        Architecture is derived from https://doi.org/10.1186/s13321-017-0232-0

        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        :param dropout: fraction of randomly disabled neurons at each epoch during training
        :param random_seed: seed of random number generators
        """
        super(MultiTaskNNClassifier, self).__init__(out, epochs, lr, early_stop, batch_size, dropout, random_seed)
        self.criterion = nn.BCELoss()
        self.activation = nn.Sigmoid()
        self.dropoutl = nn.Dropout(self.dropout)

    def set_architecture(self, n_dim: int, n_task: int):
        """Set dimension of input and number of classes to be predicted.

        :param n_dim: number of input parameters
        :param n_task: number of tasks to be predicted at the same time
        """
        if n_task < 2:
            raise ValueError('use SingleTaskNNClassifier for a single task')
        super().set_architecture([n_dim, 8000, 4000, 2000, n_task])
        cuda(self)

    def forward(self, X, istrain=False):
        """Calculate model output from input data.

        :param X: input data
        :param istrain: whether called during training, to activate dropout
        """
        input = X
        for layer in self.fcl[:-1]:
            input = F.relu(layer(input))
            if istrain:
                input = self.dropoutl(input)
        return self.activation(self.fcl[-1](input))

    def predict_proba(self, X):
        """Predict class probabilities for the incoming data

        :param X: features to predict the endpoint probabilities from
        """
        y = super().predict(X)
        return y

    def predict(self, X):
        """Predict classes for the incoming data

        :param X: features to predict the endpoint(s) from
        """
        probas = self.predict_proba(X)
        return np.round(probas)


class MultiTaskNNRegressor(BaseNN):
    def __init__(self, out: str, epochs: int = 100, lr: float = 1e-3,
                 early_stop: int = 100, batch_size: int = 1024, dropout: float = 0.25,
                 random_seed: Optional[int] = None):
        """Neural Network regressor to predict multiple endpoints.

        Architecture is adapted from https://doi.org/10.1186/s13321-017-0232-0 for multi-task regression

        :param out: output folder
        :param epochs: number of epochs
        :param lr: learning rate
        :param early_stop: stop after these many epochs without any decrease of loss
        :param batch_size: size of data batches
        :param dropout: fraction of randomly disabled neurons at each epoch during training
        :param random_seed: seed of random number generators
        """
        super(MultiTaskNNRegressor, self).__init__(out, epochs, lr, early_stop, batch_size, dropout, random_seed)
        self.dropoutl = nn.Dropout(self.dropout)
        self.criterion = nn.MSELoss()

    def set_architecture(self, n_dim: int, n_task: int):
        """Set dimension of input.

        :param n_dim: number of input parameters
        :param n_task: number of tasks to be predicted at the same time
        """
        if n_task < 2:
            raise ValueError('use SingleTaskNNRegressor for a single task')
        super().set_architecture([n_dim, 8000, 4000, 2000, n_task])
        cuda(self)

    def forward(self, X, istrain=False):
        """Calculate model output from input data.

        :param X: input data
        :param istrain: whether called during training, to activate dropout
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropoutl(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropoutl(y)
        y = self.output(y)
        return y


def loader_from_dataframe(X: pd.DataFrame,
                          Y: Union[pd.Series, pd.DataFrame],
                          batch_size: int = 1024):
    """Get PyTorch data loaders from pandas dataframes

    :param X: features to predict Y from
    :param Y: feature(s) to be predicted (dependent variable(s))
    :param batch_size: batch size of the data loader
    """
    if Y is None:
        raise ValueError('Y must be specified')
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, (pd.Series, pd.DataFrame)):
        Y = Y.values
    if len(Y.shape) == 1:
        Y = Y.reshape(Y.shape[0], 1)
    dataset = TensorDataset(T.Tensor(X), T.Tensor(Y))
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


def loader_from_iterator(X: Union[PandasTextFileReader, Iterator],
                         Y: Union[PandasTextFileReader, Iterator] = None,
                         y_col: Optional[str] = None,
                         batch_size: int = 1024):
    """Get PyTorch data loaders from iterators

    :param X: features to predict Y from
    :param Y: features to be predicted (dependent variables)
    :param y_col: name of the columns in X containing the dependent variables to be predicted
    :param batch_size: batch size of the data loader
    """
    if Y is None and y_col is None:
        raise ValueError('either Y or y_col must be specified')
    if Y is None:
        X, Y = split_into_x_and_y(X, y_col)
    dataset = IterableDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size)


class IterableDataset(PandasIterableDataset):
    def __init__(self, x_iterator: Iterator, y_iterator: Iterator):
        self.iterator = zip(x_iterator, y_iterator)

    def __iter__(self):
        for chunk_x, chunk_y in self.iterator:
            for row in zip(chunk_x, chunk_y):
                yield row


def split_into_x_and_y(data: Union[PandasTextFileReader, Iterator],
                       y_col: Union[str, List[str]]):
    """Extract the columns for the data iterator into another iterator.

    :param data: the input iterator to extract columns from
    :param y_col: name of the columns to be extracted
    :return: first iterator
    """
    if isinstance(y_col, list) and not len(y_col):
        raise ValueError('at least one column must be extracted')
    if not isinstance(y_col, list):
        y_col = [y_col]
    gen_x, gen_y = itertools.tee(data, 2)
    return extract_x(gen_x, y_col), extract_y(gen_y, y_col)


def extract_y(data: Union[PandasTextFileReader, Iterator], y_col: List[str]):
    """Extract the columns from the data."""
    for chunk in data:
        if not np.all(chunk.columns.isin(y_col)):
            raise ValueError(f'columns {chunk.columns} not found in data')
        return T.Tensor(chunk[y_col])


def extract_x(data: Union[PandasTextFileReader, Iterator], y_col: List[str]):
    """Extract the columns from the data."""
    for chunk in data:
        if not np.all(data.columns.isin(y_col)):
            raise ValueError(f'columns {chunk.columns} not found in data')
        return T.Tensor(chunk.drop(columns=y_col))
