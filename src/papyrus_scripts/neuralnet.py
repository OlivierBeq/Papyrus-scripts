# -*- coding: utf-8 -*-

"""Skorch-based neural network estimators for QSAR/PCM modelling.

Training, validation, early stopping, learning-rate decay, checkpointing and
device placement are delegated to skorch/PyTorch instead of being
hand-implemented, and every estimator exposes the same ``fit``/``predict``/
``predict_proba`` surface as an sklearn estimator (skorch's own
``NeuralNetClassifier``/``NeuralNetRegressor`` already subclass sklearn's
``ClassifierMixin``/``RegressorMixin``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

try:
    import skorch
    import torch
    from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
    from skorch.dataset import Dataset as SkorchDataset
    from skorch.helper import predefined_split
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch() -> None:
    """Raise a clear error when PyTorch/skorch, optional dependencies, are not installed."""
    if not HAS_TORCH:
        raise ImportError('Some required dependencies are missing:\n\tpytorch, skorch')


def _default_device() -> str:
    """Return the best available compute device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _set_seed(seed: int | None) -> None:
    """Seed torch/numpy RNGs and force deterministic cuDNN kernels."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _MLP(nn.Module if HAS_TORCH else object):  # type: ignore[misc]
    """Fully connected feed-forward network: ``dimensions[0] -> ... -> dimensions[-1]``.

    ReLU + dropout between hidden layers; an optional final activation
    (e.g. sigmoid/softmax) is applied to the last layer's output.
    """

    def __init__(self, dimensions: list[int], dropout: float = 0.25,
                 final_activation: nn.Module | None = None) -> None:
        super().__init__()
        self.fcl = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(len(dimensions) - 1)
        )
        self.dropoutl = nn.Dropout(dropout)
        self.final_activation = final_activation

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute model output from input data, applying dropout while training."""
        for layer in self.fcl[:-1]:
            X = torch.relu(layer(X))
            if self.training:
                X = self.dropoutl(X)
        X = self.fcl[-1](X)
        return self.final_activation(X) if self.final_activation is not None else X


class BaseNN:
    """Shared configuration for skorch-based QSAR/PCM neural network estimators.

    Mixed into :class:`skorch.NeuralNetClassifier` / :class:`skorch.NeuralNetRegressor`
    subclasses. Provides the output-folder bookkeeping, early stopping,
    checkpointing (with automatic reload of the best weights), learning-rate
    decay and reproducible seeding that were previously hand-rolled.

    Architecture is derived from https://doi.org/10.1186/s13321-017-0232-0

    Concrete subclasses must implement :meth:`set_architecture`, which builds
    the network once the input/output dimensionality is known and must be
    called before :meth:`fit`. A validation set must be supplied via
    :meth:`set_validation` before :meth:`fit` as well — mirroring the
    previous requirement, but raising a clear error instead of failing deep
    inside the training loop.
    """

    def __init__(self, out: str | Path, epochs: int = 100, lr: float = 1e-3,
                 early_stop: int = 100, batch_size: int = 1024, dropout: float = 0.25,
                 random_seed: int | None = None, **kwargs) -> None:
        """Configure the estimator.

        :param out: output folder for checkpoints (best weights, optimizer
            and criterion state) and the training history
        :param epochs: maximum number of epochs
        :param lr: initial learning rate
        :param early_stop: stop after these many epochs without any decrease
            of the validation loss
        :param batch_size: size of data batches
        :param dropout: fraction of randomly disabled neurons at each epoch
            during training
        :param random_seed: seed of random number generators
        """
        _require_torch()
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dropout = dropout
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed) if random_seed is not None else None
        self._dims: list[int] | None = None
        _set_seed(random_seed)

        callbacks = [
            ('early_stop', EarlyStopping(patience=early_stop)),
            ('checkpoint', Checkpoint(
                dirname=str(self.out), f_history='training_history.json', load_best=True,
            )),
            ('lr_scheduler', LRScheduler(
                policy=torch.optim.lr_scheduler.LambdaLR,
                # Reproduces the original hand-rolled decay: lr * (1 - 1/epochs) ** (epoch * 10)
                lr_lambda=lambda epoch: (1 - 1 / epochs) ** (epoch * 10),
            )),
        ]
        # BaseNN is a mixin: only meaningful combined with a skorch NeuralNet
        # subclass (see SingleTaskNNClassifier etc.), which supplies the
        # rest of this __init__ signature and the fit/predict_proba/
        # initialize members used below - invisible to mypy from here.
        super().__init__(  # type: ignore[call-arg]
            module=_MLP,
            optimizer=torch.optim.Adam,
            lr=lr,
            max_epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            callbacks__valid_acc=None,  # replaced by our own early-stopping/checkpoint logic
            predict_nonlinearity=None,  # the module's own final_activation already applies one
            device=_default_device(),
            train_split=None,  # require an explicit validation set, see set_validation()
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Architecture / data preparation
    # ------------------------------------------------------------------

    def initialize_module(self) -> BaseNN:
        """Build the network now that :meth:`set_architecture` has recorded its shape."""
        if self._dims is None:
            raise ValueError('call set_architecture() before fit()')
        self.module_ = _MLP(self._dims, dropout=self.dropout,
                             final_activation=self._build_final_activation())
        return self

    def _build_final_activation(self) -> nn.Module | None:
        """Return the final-layer activation module; overridden by subclasses as needed."""
        return None

    @staticmethod
    def _prep_X(X: pd.DataFrame | pl.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        return np.asarray(X, dtype='float32')

    def _prep_y(self, y: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
        """Coerce *y* into the dtype/shape expected by ``self.criterion``. Overridable."""
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        y = np.asarray(y, dtype='float32')
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def set_validation(self, X: pd.DataFrame | np.ndarray, y: pd.Series | pd.DataFrame | np.ndarray) -> None:
        """Set the validation set to be used during fitting (required before :meth:`fit`).

        Must be called after :meth:`set_architecture`, which determines how
        *y* needs to be encoded (e.g. class indices vs. one-hot probabilities).

        :param X: features to predict y from
        :param y: feature(s) to be predicted (dependent variable(s))
        """
        self.train_split = predefined_split(SkorchDataset(self._prep_X(X), self._prep_y(y)))

    # ------------------------------------------------------------------
    # sklearn-style estimator interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | pd.DataFrame | np.ndarray, **kwargs) -> BaseNN:
        """Fit the network on the training set, validating against the set from :meth:`set_validation`.

        :param X: features to predict y from
        :param y: feature(s) to be predicted (dependent variable(s))
        """
        if self.train_split is None:
            raise ValueError('call set_validation(X, y) before fit()')
        return super().fit(self._prep_X(X), self._prep_y(y), **kwargs)  # type: ignore[misc]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class/task probabilities for the incoming data.

        :param X: features to predict the endpoint probabilities from
        """
        return super().predict_proba(self._prep_X(X))  # type: ignore[misc]

    def reset(self) -> None:
        """Reset weights, optimizer and criterion to a freshly initialised state."""
        self.initialize()  # type: ignore[attr-defined]


class SingleTaskNNClassifier(BaseNN, skorch.NeuralNetClassifier if HAS_TORCH else object):  # type: ignore[misc]
    """Neural Network classifier to predict a unique endpoint."""

    def set_architecture(self, n_dim: int, n_class: int) -> None:
        """Set dimension of input and number of classes to be predicted.

        :param n_dim: number of input parameters
        :param n_class: number of one-hot encoded classes (i.e. 1 for binary endpoint not one-hot encoded)
        """
        if n_class < 1:
            raise ValueError('can only perform binary (n_class=1 or n_class=2)'
                             ' or multi-classes predictions (n_class>2)')
        self._n_classes_ = n_class
        self._n_features_in_ = n_dim
        self._dims = [n_dim, 8000, 4000, 2000, n_class]
        # Binary: sigmoid + BCE on independent probabilities.
        # Multi-class: raw logits + cross-entropy (softmax applied internally by
        # the loss, and by skorch's predict_proba via predict_nonlinearity='auto').
        self.criterion = nn.BCELoss if n_class == 1 else nn.CrossEntropyLoss
        self.predict_nonlinearity = None if n_class == 1 else 'auto'

    def _build_final_activation(self) -> nn.Module | None:
        return nn.Sigmoid() if self._n_classes_ == 1 else None

    def _prep_y(self, y: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
        if not hasattr(self, '_n_classes_'):
            raise ValueError('call set_architecture() before set_validation() or fit()')
        if self._n_classes_ > 1:
            y = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
            return y.astype('int64').reshape(-1)
        return super()._prep_y(y)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities for the incoming data.

        :param X: features to predict the endpoint probabilities from
        """
        return super().predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict classes for the incoming data.

        Binary endpoints threshold the sigmoid output at 0.5; multi-class
        endpoints use the highest-probability class (standard argmax).

        :param X: features to predict the endpoint(s) from
        """
        if self._n_classes_ == 1:
            return np.round(self.predict_proba(X))
        return super().predict(self._prep_X(X))  # type: ignore[misc]


class SingleTaskNNRegressor(BaseNN, skorch.NeuralNetRegressor if HAS_TORCH else object):  # type: ignore[misc]
    """Neural Network regressor to predict a unique endpoint."""

    def __init__(self, *args, **kwargs) -> None:
        """Neural Network regressor to predict a unique endpoint."""
        super().__init__(*args, criterion=nn.MSELoss, **kwargs)

    def set_architecture(self, n_dim: int) -> None:
        """Set dimension of input.

        :param n_dim: number of input parameters
        """
        self._dims = [n_dim, 8000, 4000, 2000, 1]


class MultiTaskNNClassifier(BaseNN, skorch.NeuralNetClassifier if HAS_TORCH else object):  # type: ignore[misc]
    """Neural Network classifier to predict multiple (independent, binary) endpoints."""

    def __init__(self, *args, **kwargs) -> None:
        """Neural Network classifier to predict multiple endpoints."""
        super().__init__(*args, criterion=nn.BCELoss, **kwargs)

    def set_architecture(self, n_dim: int, n_task: int) -> None:
        """Set dimension of input and number of tasks to be predicted.

        :param n_dim: number of input parameters
        :param n_task: number of tasks to be predicted at the same time
        """
        if n_task < 2:
            raise ValueError('use SingleTaskNNClassifier for a single task')
        self._dims = [n_dim, 8000, 4000, 2000, n_task]

    def _build_final_activation(self) -> nn.Module:
        return nn.Sigmoid()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict per-task probabilities for the incoming data.

        :param X: features to predict the endpoint probabilities from
        """
        return super().predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict per-task classes (each output thresholded independently at 0.5).

        :param X: features to predict the endpoint(s) from
        """
        return np.round(self.predict_proba(X))


class MultiTaskNNRegressor(BaseNN, skorch.NeuralNetRegressor if HAS_TORCH else object):  # type: ignore[misc]
    """Neural Network regressor to predict multiple endpoints."""

    def __init__(self, *args, **kwargs) -> None:
        """Neural Network regressor to predict multiple endpoints."""
        super().__init__(*args, criterion=nn.MSELoss, **kwargs)

    def set_architecture(self, n_dim: int, n_task: int) -> None:
        """Set dimension of input and number of tasks to be predicted.

        :param n_dim: number of input parameters
        :param n_task: number of tasks to be predicted at the same time
        """
        if n_task < 2:
            raise ValueError('use SingleTaskNNRegressor for a single task')
        self._dims = [n_dim, 8000, 4000, 2000, n_task]
