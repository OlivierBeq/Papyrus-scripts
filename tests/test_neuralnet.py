# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.neuralnet.

Trains tiny networks (few epochs, small random data) purely to exercise the
skorch wiring - not to check model quality. Skipped when the optional
torch/skorch dependencies are not installed.
"""

import functools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from src.papyrus_scripts import neuralnet as nn_mod
from src.papyrus_scripts.neuralnet import BaseNN

TORCH_AVAILABLE = nn_mod.HAS_TORCH

if TORCH_AVAILABLE:
    from skorch.callbacks import Callback

    from src.papyrus_scripts.neuralnet import (
        MultiTaskNNClassifier,
        MultiTaskNNRegressor,
        SingleTaskNNClassifier,
        SingleTaskNNRegressor,
    )


def _rng():
    return np.random.default_rng(0)


_mps_patcher = None


def setUpModule():
    # Force CPU everywhere fit/predict actually run a model: the MPS backend
    # is prone to segfaulting on headless CI runners (no crash on the
    # training/backward pass, only on eval-mode forward passes), which isn't
    # something these wiring tests need to exercise. TestDefaultDevice below
    # patches is_available() itself to test the pure selection logic, so it
    # is unaffected by this module-wide override.
    global _mps_patcher
    if TORCH_AVAILABLE:
        # lru_cache-wrapped: dynamo reads __wrapped__, real is_available has it too.
        _mps_patcher = patch.object(
            nn_mod.torch.backends.mps, 'is_available',
            new=functools.lru_cache(maxsize=None)(lambda: False),
        )
        _mps_patcher.start()


def tearDownModule():
    if _mps_patcher is not None:
        _mps_patcher.stop()


if TORCH_AVAILABLE:
    class _LossRecorder(Callback):
        """Records train_loss every epoch (net.history gets truncated on reload)."""

        def __init__(self):
            self.losses: list[float] = []

        def on_epoch_end(self, net, **kwargs):
            self.losses.append(net.history[-1, 'train_loss'])


def _ill_conditioned_regression_data(n_train=160, n_valid=40, seed=0):
    """Regression data: raw features span ~9 orders of magnitude, target stays small."""
    rng = np.random.default_rng(seed)
    n = n_train + n_valid
    z = rng.standard_normal((n, 4))
    y = z @ np.array([1.0, 1.0, 1.0, 1.0]) + rng.normal(0, 0.1, n)
    scales = np.array([1.0, 1e4, 1e-3, 1e6])
    offsets = np.array([0.0, 5e4, 2e-3, -3e6])
    X = z * scales + offsets
    return (pd.DataFrame(X[:n_train]), pd.Series(y[:n_train]),
            pd.DataFrame(X[n_train:]), pd.Series(y[n_train:]))


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestFeatureScaling(unittest.TestCase):
    """BaseNN standardizes inputs by default (see neuralnet.py's `scale` param)."""

    def _fit_regressor(self, X, y, X_valid, y_valid, *, scale, tmpdir):
        recorder = _LossRecorder()
        reg = SingleTaskNNRegressor(tmpdir, epochs=60, early_stop=60, lr=0.01,
                                    hidden_layers=[16, 8], scale=scale, random_seed=0, verbose=0)
        reg.set_architecture(4)
        reg.set_validation(X_valid, y_valid)
        reg.callbacks.append(('loss_recorder', recorder))
        reg.fit(X, y)
        return reg, recorder.losses

    def test_training_loss_decreases_when_features_are_scaled(self):
        X, y, X_valid, y_valid = _ill_conditioned_regression_data()
        with tempfile.TemporaryDirectory() as d:
            _, losses = self._fit_regressor(X, y, X_valid, y_valid, scale=True, tmpdir=d)
        self.assertLess(losses[-1], losses[0] * 0.9)

    def test_unscaled_features_leave_loss_orders_of_magnitude_larger(self):
        X, y, X_valid, y_valid = _ill_conditioned_regression_data()
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            _, scaled_losses = self._fit_regressor(X, y, X_valid, y_valid, scale=True, tmpdir=d1)
            _, unscaled_losses = self._fit_regressor(X, y, X_valid, y_valid, scale=False, tmpdir=d2)
        self.assertLess(scaled_losses[-1], unscaled_losses[-1] / 1e6)

    def test_scaler_is_fit_on_training_data_only(self):
        X, y, X_valid, y_valid = _ill_conditioned_regression_data()
        with tempfile.TemporaryDirectory() as d:
            reg, _ = self._fit_regressor(X, y, X_valid, y_valid, scale=True, tmpdir=d)
        expected_mean = X.to_numpy(dtype='float32').mean(axis=0)
        expected_scale = X.to_numpy(dtype='float32').std(axis=0)
        np.testing.assert_allclose(reg.scaler_.mean_, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(reg.scaler_.scale_, expected_scale, rtol=1e-5)

    def test_scale_false_disables_scaler(self):
        X, y, X_valid, y_valid = _ill_conditioned_regression_data()
        with tempfile.TemporaryDirectory() as d:
            reg, _ = self._fit_regressor(X, y, X_valid, y_valid, scale=False, tmpdir=d)
        self.assertIsNone(reg.scaler_)

    def test_predict_applies_the_fitted_scaler(self):
        X, y, X_valid, y_valid = _ill_conditioned_regression_data()
        with tempfile.TemporaryDirectory() as d:
            reg, _ = self._fit_regressor(X, y, X_valid, y_valid, scale=True, tmpdir=d)
            X_scaled = reg.scaler_.transform(X_valid.to_numpy(dtype='float32')).astype('float32')
            with nn_mod.torch.no_grad():
                reg.module_.eval()
                expected = reg.module_(nn_mod.torch.from_numpy(X_scaled)).numpy()
            actual = reg.predict(X_valid)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


class TestPrepX(unittest.TestCase):
    """BaseNN._prep_X needs no torch/skorch - pure numpy/pandas/polars logic."""

    def test_polars_dataframe_matches_pandas_equivalent(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        from_pandas = BaseNN._prep_X(pd.DataFrame(data))
        from_polars = BaseNN._prep_X(pl.DataFrame(data, schema=['0', '1'], orient='row'))
        np.testing.assert_array_equal(from_polars, from_pandas)
        self.assertEqual(from_polars.dtype, np.dtype('float32'))


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestRequireTorch(unittest.TestCase):

    def test_raises_when_torch_unavailable(self):
        with patch.object(nn_mod, 'HAS_TORCH', False):
            with self.assertRaises(ImportError):
                nn_mod._require_torch()

    def test_does_not_raise_when_torch_available(self):
        nn_mod._require_torch()


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestDefaultDevice(unittest.TestCase):

    def test_prefers_cuda(self):
        with patch.object(nn_mod.torch.cuda, 'is_available', return_value=True):
            self.assertEqual(nn_mod._default_device(), 'cuda')

    def test_falls_back_to_mps(self):
        with (
            patch.object(nn_mod.torch.cuda, 'is_available', return_value=False),
            patch.object(nn_mod.torch.backends.mps, 'is_available', return_value=True),
        ):
            self.assertEqual(nn_mod._default_device(), 'mps')

    def test_falls_back_to_cpu(self):
        with (
            patch.object(nn_mod.torch.cuda, 'is_available', return_value=False),
            patch.object(nn_mod.torch.backends.mps, 'is_available', return_value=False),
        ):
            self.assertEqual(nn_mod._default_device(), 'cpu')


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestDeviceOverride(unittest.TestCase):
    """Regression: passing device= used to collide with the hardcoded default."""

    def test_explicit_device_kwarg_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as d:
            reg = SingleTaskNNRegressor(d, epochs=2, device='cpu')
        self.assertEqual(reg.device, 'cpu')

    def test_omitting_device_still_defaults_via_default_device(self):
        with tempfile.TemporaryDirectory() as d:
            reg = SingleTaskNNRegressor(d, epochs=2)
        self.assertEqual(reg.device, nn_mod._default_device())

    @unittest.skipIf(nn_mod.torch.cuda.is_available(), 'meaningful only when no GPU is present')
    def test_requesting_cuda_when_unavailable_raises_at_fit(self):
        # must fail loudly at fit(), not silently fall back to CPU
        rng = _rng()
        X, y = pd.DataFrame(rng.random((12, 4))), pd.Series(rng.random(12))
        X_valid, y_valid = pd.DataFrame(rng.random((4, 4))), pd.Series(rng.random(4))
        with tempfile.TemporaryDirectory() as d:
            reg = SingleTaskNNRegressor(d, epochs=2, early_stop=2, hidden_layers=[4],
                                        device='cuda', verbose=0)
            reg.set_architecture(4)
            reg.set_validation(X_valid, y_valid)
            with self.assertRaises((RuntimeError, AssertionError)):
                reg.fit(X, y)

    @unittest.skipIf(nn_mod.torch.cuda.is_available(), 'meaningful only when no GPU is present')
    def test_set_params_cuda_after_construction_raises_at_fit(self):
        rng = _rng()
        X, y = pd.DataFrame(rng.random((12, 4))), pd.Series(rng.random(12))
        X_valid, y_valid = pd.DataFrame(rng.random((4, 4))), pd.Series(rng.random(4))
        with tempfile.TemporaryDirectory() as d:
            reg = SingleTaskNNRegressor(d, epochs=2, early_stop=2, hidden_layers=[4], verbose=0)
            reg.set_params(device='cuda')
            reg.set_architecture(4)
            reg.set_validation(X_valid, y_valid)
            with self.assertRaises((RuntimeError, AssertionError)):
                reg.fit(X, y)


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestBatchSize(unittest.TestCase):

    def _fit(self, *, batch_size, n_train, tmpdir):
        rng = _rng()
        X = pd.DataFrame(rng.random((n_train, 4)))
        y = pd.Series(rng.random(n_train))
        X_valid = pd.DataFrame(rng.random((4, 4)))
        y_valid = pd.Series(rng.random(4))
        reg = SingleTaskNNRegressor(tmpdir, epochs=2, early_stop=2, hidden_layers=[4],
                                    batch_size=batch_size, verbose=0)
        reg.set_architecture(4)
        reg.set_validation(X_valid, y_valid)
        reg.fit(X, y)
        return reg, X

    def test_batch_size_of_one_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            reg, X = self._fit(batch_size=1, n_train=8, tmpdir=d)
        self.assertEqual(reg.predict(X).shape, (8, 1))

    def test_batch_size_larger_than_training_set_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            reg, X = self._fit(batch_size=1024, n_train=8, tmpdir=d)
        self.assertEqual(reg.predict(X).shape, (8, 1))


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestTinyTrainingSet(unittest.TestCase):

    def test_two_training_and_two_validation_samples_does_not_raise(self):
        rng = _rng()
        X = pd.DataFrame(rng.random((2, 4)))
        y = pd.Series(rng.random(2))
        X_valid = pd.DataFrame(rng.random((2, 4)))
        y_valid = pd.Series(rng.random(2))
        with tempfile.TemporaryDirectory() as d:
            reg = SingleTaskNNRegressor(d, epochs=2, early_stop=2, batch_size=1,
                                        hidden_layers=[4], verbose=0)
            reg.set_architecture(4)
            reg.set_validation(X_valid, y_valid)
            reg.fit(X, y)
            preds = reg.predict(X)
        self.assertEqual(preds.shape, (2, 1))


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestSetSeed(unittest.TestCase):

    def test_none_seed_is_a_no_op(self):
        nn_mod._set_seed(None)

    def test_seed_sets_deterministic_cudnn_flags(self):
        nn_mod._set_seed(42)
        self.assertTrue(nn_mod.torch.backends.cudnn.deterministic)
        self.assertFalse(nn_mod.torch.backends.cudnn.benchmark)


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestSingleTaskNNClassifierBinary(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        rng = _rng()
        self.X = pd.DataFrame(rng.random((40, 4)))
        self.y = pd.Series(rng.integers(0, 2, 40))
        self.X_valid = pd.DataFrame(rng.random((10, 4)))
        self.y_valid = pd.Series(rng.integers(0, 2, 10))

    def tearDown(self):
        self._tmpdir.cleanup()

    def _fitted(self, **kwargs):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2, early_stop=2, lr=0.01,
                                     hidden_layers=[8, 4], **kwargs)
        clf.set_architecture(4, 1)
        clf.set_validation(self.X_valid, self.y_valid)
        clf.fit(self.X, self.y)
        return clf

    def test_predict_returns_binary_labels(self):
        clf = self._fitted()
        preds = clf.predict(self.X)
        self.assertEqual(preds.shape, (40, 1))
        self.assertTrue(set(np.unique(preds)).issubset({0.0, 1.0}))

    def test_predict_proba_shape(self):
        clf = self._fitted()
        proba = clf.predict_proba(self.X)
        self.assertEqual(proba.shape, (40, 1))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_classes_(self):
        clf = self._fitted()
        self.assertEqual(sorted(clf.classes_), [0.0, 1.0])

    def test_checkpoint_files_written_to_out(self):
        self._fitted()
        written = {p.name for p in Path(self._tmpdir.name).iterdir()}
        self.assertIn('params.pt', written)
        self.assertIn('training_history.json', written)

    def test_reset_reinitialises_weights(self):
        clf = self._fitted()
        before = clf.module_.fcl[0].weight.clone()
        clf.reset()
        after = clf.module_.fcl[0].weight
        self.assertFalse((before == after).all().item())

    def test_fit_without_set_validation_raises(self):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2)
        clf.set_architecture(4, 1)
        with self.assertRaises(ValueError):
            clf.fit(self.X, self.y)

    def test_set_validation_without_set_architecture_raises(self):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2)
        with self.assertRaises(ValueError):
            clf.set_validation(self.X_valid, self.y_valid)

    def test_fit_without_set_architecture_raises(self):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2)
        with self.assertRaises(ValueError):
            clf.fit(self.X, self.y)

    def test_initialize_module_without_set_architecture_raises(self):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2)
        with self.assertRaises(ValueError):
            clf.initialize_module()

    def test_set_architecture_rejects_non_positive_n_class(self):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2)
        with self.assertRaises(ValueError):
            clf.set_architecture(4, 0)

    def test_custom_hidden_layers_are_used(self):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2, hidden_layers=[16, 8])
        clf.set_architecture(4, 1)
        self.assertEqual(clf._dims, [4, 16, 8, 1])

    def test_default_hidden_layers_unchanged(self):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2)
        clf.set_architecture(4, 1)
        self.assertEqual(clf._dims, [4, 8000, 4000, 2000, 1])

    def test_uses_bce_with_logits_and_raw_logits_module(self):
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2, hidden_layers=[8, 4])
        clf.set_architecture(4, 1)
        self.assertIs(clf.criterion, nn_mod.nn.BCEWithLogitsLoss)
        self.assertIs(clf.predict_nonlinearity, nn_mod.torch.sigmoid)
        clf.initialize()
        self.assertIsNone(clf.module_.final_activation)


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestSingleTaskNNClassifierMultiClass(unittest.TestCase):

    def test_predict_uses_argmax_over_classes(self):
        rng = _rng()
        X = pd.DataFrame(rng.random((40, 4)))
        y = pd.Series(rng.integers(0, 3, 40))
        X_valid = pd.DataFrame(rng.random((10, 4)))
        y_valid = pd.Series(rng.integers(0, 3, 10))
        with tempfile.TemporaryDirectory() as d:
            clf = SingleTaskNNClassifier(d, epochs=2, early_stop=2, lr=0.01, hidden_layers=[8, 4])
            clf.set_architecture(4, 3)
            clf.set_validation(X_valid, y_valid)
            clf.fit(X, y)
            preds = clf.predict(X)
            proba = clf.predict_proba(X)
            self.assertEqual(preds.shape, (40,))
            self.assertEqual(proba.shape, (40, 3))
            self.assertTrue(set(np.unique(preds)).issubset({0, 1, 2}))


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestSingleTaskNNRegressor(unittest.TestCase):

    def test_predict_shape(self):
        rng = _rng()
        X = pd.DataFrame(rng.random((40, 4)))
        y = pd.Series(rng.random(40))
        X_valid = pd.DataFrame(rng.random((10, 4)))
        y_valid = pd.Series(rng.random(10))
        with tempfile.TemporaryDirectory() as d:
            reg = SingleTaskNNRegressor(d, epochs=2, early_stop=2, lr=0.01, hidden_layers=[8, 4])
            reg.set_architecture(4)
            reg.set_validation(X_valid, y_valid)
            reg.fit(X, y)
            preds = reg.predict(X)
            self.assertEqual(preds.shape, (40, 1))


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestMultiTaskNN(unittest.TestCase):

    def test_classifier_predicts_independent_binary_tasks(self):
        rng = _rng()
        X = pd.DataFrame(rng.random((40, 4)))
        y = pd.DataFrame(rng.integers(0, 2, (40, 3)))
        X_valid = pd.DataFrame(rng.random((10, 4)))
        y_valid = pd.DataFrame(rng.integers(0, 2, (10, 3)))
        with tempfile.TemporaryDirectory() as d:
            clf = MultiTaskNNClassifier(d, epochs=2, early_stop=2, lr=0.01, hidden_layers=[8, 4])
            clf.set_architecture(4, 3)
            clf.set_validation(X_valid, y_valid)
            clf.fit(X, y)
            preds = clf.predict(X)
            self.assertEqual(preds.shape, (40, 3))
            self.assertTrue(set(np.unique(preds)).issubset({0.0, 1.0}))

    def test_classifier_requires_at_least_two_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            clf = MultiTaskNNClassifier(d, epochs=2)
            with self.assertRaises(ValueError):
                clf.set_architecture(4, 1)

    def test_uses_bce_with_logits_and_raw_logits_module(self):
        with tempfile.TemporaryDirectory() as d:
            clf = MultiTaskNNClassifier(d, epochs=2, hidden_layers=[8, 4])
            clf.set_architecture(4, 3)
            self.assertIs(clf.criterion, nn_mod.nn.BCEWithLogitsLoss)
            self.assertIs(clf.predict_nonlinearity, nn_mod.torch.sigmoid)
            clf.initialize()
            self.assertIsNone(clf.module_.final_activation)

    def test_regressor_predicts_multiple_tasks(self):
        rng = _rng()
        X = pd.DataFrame(rng.random((40, 4)))
        y = pd.DataFrame(rng.random((40, 3)))
        X_valid = pd.DataFrame(rng.random((10, 4)))
        y_valid = pd.DataFrame(rng.random((10, 3)))
        with tempfile.TemporaryDirectory() as d:
            reg = MultiTaskNNRegressor(d, epochs=2, early_stop=2, lr=0.01, hidden_layers=[8, 4])
            reg.set_architecture(4, 3)
            reg.set_validation(X_valid, y_valid)
            reg.fit(X, y)
            preds = reg.predict(X)
            self.assertEqual(preds.shape, (40, 3))

    def test_regressor_requires_at_least_two_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            reg = MultiTaskNNRegressor(d, epochs=2)
            with self.assertRaises(ValueError):
                reg.set_architecture(4, 1)


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestMultiTaskMaskedLoss(unittest.TestCase):
    """Multi-task data is rarely dense - NaN targets must be excluded from the loss, not crash it."""

    def test_criterion_reduction_is_none(self):
        # _MaskedMultiTaskLoss.get_loss needs an unreduced, per-element loss to mask.
        with tempfile.TemporaryDirectory() as d:
            reg = MultiTaskNNRegressor(d, epochs=2)
            clf = MultiTaskNNClassifier(d, epochs=2)
        self.assertEqual(reg.criterion__reduction, 'none')
        self.assertEqual(clf.criterion__reduction, 'none')

    def test_regressor_loss_ignores_nan_targets(self):
        with tempfile.TemporaryDirectory() as d:
            reg = MultiTaskNNRegressor(d, epochs=1)
            reg.set_architecture(4, 3)
            reg.initialize()
            y_pred = nn_mod.torch.tensor([[1.0, 2.0, 3.0]])
            y_true = nn_mod.torch.tensor([[1.0, float('nan'), 5.0]])
            loss = reg.get_loss(y_pred, y_true.numpy())
        # column 1 (NaN) excluded: mean((1-1)**2, (3-5)**2) = 2.0
        self.assertAlmostEqual(loss.item(), 2.0, places=5)

    def test_all_nan_batch_does_not_raise_or_produce_nan_loss(self):
        with tempfile.TemporaryDirectory() as d:
            reg = MultiTaskNNRegressor(d, epochs=1)
            reg.set_architecture(4, 3)
            reg.initialize()
            y_pred = nn_mod.torch.zeros((2, 3))
            y_true = nn_mod.torch.full((2, 3), float('nan'))
            loss = reg.get_loss(y_pred, y_true.numpy())
        self.assertEqual(loss.item(), 0.0)

    def test_regressor_fits_with_sparse_targets(self):
        rng = _rng()
        X = pd.DataFrame(rng.random((40, 4)))
        y = pd.DataFrame(rng.random((40, 3)))
        y.iloc[::2, 1] = np.nan  # half of task 1's labels missing
        X_valid = pd.DataFrame(rng.random((10, 4)))
        y_valid = pd.DataFrame(rng.random((10, 3)))
        y_valid.iloc[::3, 0] = np.nan
        with tempfile.TemporaryDirectory() as d:
            reg = MultiTaskNNRegressor(d, epochs=2, early_stop=2, lr=0.01, hidden_layers=[8, 4])
            reg.set_architecture(4, 3)
            reg.set_validation(X_valid, y_valid)
            reg.fit(X, y)
            preds = reg.predict(X)
        self.assertEqual(preds.shape, (40, 3))
        self.assertFalse(np.isnan(preds).any())

    def test_classifier_fits_with_sparse_targets(self):
        rng = _rng()
        X = pd.DataFrame(rng.random((40, 4)))
        y = pd.DataFrame(rng.integers(0, 2, (40, 3)).astype(float))
        y.iloc[::2, 1] = np.nan
        X_valid = pd.DataFrame(rng.random((10, 4)))
        y_valid = pd.DataFrame(rng.integers(0, 2, (10, 3)).astype(float))
        y_valid.iloc[::3, 0] = np.nan
        with tempfile.TemporaryDirectory() as d:
            clf = MultiTaskNNClassifier(d, epochs=2, early_stop=2, lr=0.01, hidden_layers=[8, 4])
            clf.set_architecture(4, 3)
            clf.set_validation(X_valid, y_valid)
            clf.fit(X, y)
            preds = clf.predict(X)
        self.assertEqual(preds.shape, (40, 3))
        self.assertTrue(set(np.unique(preds)).issubset({0.0, 1.0}))


if __name__ == '__main__':
    unittest.main()
