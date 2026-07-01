# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.neuralnet.

Trains tiny networks (few epochs, small random data) purely to exercise the
skorch wiring - not to check model quality. Skipped when the optional
torch/skorch dependencies are not installed.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.papyrus_scripts import neuralnet as nn_mod

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
        clf = SingleTaskNNClassifier(self._tmpdir.name, epochs=2, early_stop=2, lr=0.01, **kwargs)
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


@unittest.skipUnless(TORCH_AVAILABLE, 'requires torch and skorch')
class TestSingleTaskNNClassifierMultiClass(unittest.TestCase):

    def test_predict_uses_argmax_over_classes(self):
        rng = _rng()
        X = pd.DataFrame(rng.random((40, 4)))
        y = pd.Series(rng.integers(0, 3, 40))
        X_valid = pd.DataFrame(rng.random((10, 4)))
        y_valid = pd.Series(rng.integers(0, 3, 10))
        with tempfile.TemporaryDirectory() as d:
            clf = SingleTaskNNClassifier(d, epochs=2, early_stop=2, lr=0.01)
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
            reg = SingleTaskNNRegressor(d, epochs=2, early_stop=2, lr=0.01)
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
            clf = MultiTaskNNClassifier(d, epochs=2, early_stop=2, lr=0.01)
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
            reg = MultiTaskNNRegressor(d, epochs=2, early_stop=2, lr=0.01)
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
