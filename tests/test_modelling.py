# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.modelling helper functions that don't
require training a real model (no xgboost/torch fitting, no network).
"""

import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.papyrus_scripts.modelling import model_metrics, train_test_proportional_group_split


class TestTrainTestProportionalGroupSplit(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({'x': range(20)})
        self.groups = [i % 5 for i in range(20)]  # 5 groups of 4 rows each

    def test_does_not_raise(self):
        # Regression test: np.where() used to be given a bare generator
        # expression instead of a boolean array/list, raising ValueError
        # on every call - this was the sole implementation behind
        # split_by='cluster'/'custom-cluster' in qsar()/pcm().
        train_data, test_data, t_groups, best_groups = train_test_proportional_group_split(
            self.data, self.groups, test_size=0.3,
        )
        self.assertEqual(len(train_data) + len(test_data), len(self.data))

    def test_train_test_groups_are_disjoint(self):
        _, _, t_groups, best_groups = train_test_proportional_group_split(
            self.data, self.groups, test_size=0.3,
        )
        self.assertTrue(set(t_groups).isdisjoint(set(best_groups)))

    def test_split_approximates_requested_test_size(self):
        train_data, test_data, _, _ = train_test_proportional_group_split(
            self.data, self.groups, test_size=0.2,
        )
        # 5 equal-size groups: the closest achievable proportion to 0.2 is
        # exactly one group (4/20 = 0.2).
        self.assertEqual(len(test_data), 4)
        self.assertEqual(len(train_data), 16)


class TestModelMetricsMCC(unittest.TestCase):
    """Regression tests: model_metrics wrapped MCC computation in a
    try/except RuntimeWarning that never fired (matthews_corrcoef warns via
    warnings.warn and returns 0.0 for degenerate input, it never raises), so
    the except branch was dead code with no effect either way.
    """

    def test_binary_classification_mcc_present(self):
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 1, 0, 1])
        model = DecisionTreeClassifier().fit(X, y)
        values = model_metrics(model, y, X)
        self.assertIn('MCC', values)

    def test_multiclass_degenerate_single_class_does_not_raise(self):
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 0, 0])
        model = DecisionTreeClassifier().fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            values = model_metrics(model, y, X)
        self.assertIn('0|MCC', values)
        self.assertEqual(values['0|MCC'], 0.0)


if __name__ == '__main__':
    unittest.main()
