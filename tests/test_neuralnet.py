# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.neuralnet.

Trains tiny networks (few epochs, small random data) purely to exercise the
skorch wiring - not to check model quality. Skipped when the optional
torch/skorch dependencies are not installed.
"""

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
        _mps_patcher = patch.object(nn_mod.torch.backends.mps, 'is_available', return_value=False)
        _mps_patcher.start()


def tearDownModule():
    if _mps_patcher is not None:
        _mps_patcher.stop()


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


if __name__ == '__main__':
    unittest.main()
