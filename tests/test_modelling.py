# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.modelling helper functions that don't
require training a real model (no xgboost/torch fitting, no network).
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.papyrus_scripts.modelling import (
    filter_molecular_descriptors,
    model_metrics,
    qsar,
    train_test_proportional_group_split,
)


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


class TestFilterMolecularDescriptors(unittest.TestCase):
    """filter_molecular_descriptors must accept pl.DataFrame/pl.LazyFrame
    (read_molecular_descriptors' actual return type) as well as pandas, and
    always return a pandas DataFrame for qsar()/pcm()'s pandas-only logic.
    """

    def test_pandas_dataframe_input(self):
        df = pd.DataFrame({'connectivity': ['C1', 'C2', 'C3'], 'value': [1, 2, 3]})
        result = filter_molecular_descriptors(df, 'connectivity', ['C1', 'C3'])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(sorted(result['connectivity']), ['C1', 'C3'])

    def test_polars_dataframe_input_returns_pandas(self):
        df = pl.DataFrame({'connectivity': ['C1', 'C2', 'C3'], 'value': [1, 2, 3]})
        result = filter_molecular_descriptors(df, 'connectivity', ['C1', 'C3'])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(sorted(result['connectivity']), ['C1', 'C3'])

    def test_polars_lazyframe_input_returns_pandas(self):
        # read_molecular_descriptors returns exactly this by default
        # (chunksize is not None).
        lf = pl.LazyFrame({'connectivity': ['C1', 'C2', 'C3'], 'value': [1, 2, 3]})
        result = filter_molecular_descriptors(lf, 'connectivity', ['C1', 'C3'])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(sorted(result['connectivity']), ['C1', 'C3'])


def _make_bioactivity_data():
    # Small, but crosses every qsar() threshold (num_points, temporal
    # split, per-class counts, regressor amplitude) to reach a real fit,
    # not an early "insufficient data" return.
    n = 16
    return {
        'connectivity': [f'C{i}' for i in range(n)],
        'target_id': ['T1'] * n,
        'Activity_class': [None] * n,
        # 4 inactive (<=6.5) + 4 active (>6.5) on each side of the split.
        'pchembl_value_Mean': [5.0, 5.2, 5.4, 5.6, 7.0, 7.2, 7.4, 7.6] * 2,
        'relation': ['='] * n,
        'Year': [2010] * 8 + [2016] * 8,
    }


def _make_descriptors():
    n = 16
    return {'connectivity': [f'C{i}' for i in range(n)], 'Desc_1': [i / n for i in range(n)]}


class TestQsarAcceptsPolarsInput(unittest.TestCase):
    """qsar()'s internals are pandas-only (.isna(), .loc, .merge(), .iloc,
    ...), so it must accept and convert the pl.DataFrame
    PapyrusDataset.aggregate() returns.
    """

    def _run_qsar(self, data, model=None):
        # descriptor_chunksize left at its non-None default (50000), so
        # read_molecular_descriptors would really return a pl.LazyFrame -
        # mocked here with one instead of hitting the network/filesystem.
        with patch(
            'src.papyrus_scripts.modelling.read_molecular_descriptors',
            return_value=pl.LazyFrame(_make_descriptors()),
        ):
            return qsar(data, model=model or DecisionTreeClassifier(), num_points=3, folds=2,
                       split_year=2016, verbose=False)

    def test_polars_dataframe_input_does_not_raise(self):
        results, _ = self._run_qsar(pl.DataFrame(_make_bioactivity_data()))
        self.assertEqual(list(results.index.get_level_values('target').unique()), ['T1'])

    def test_pandas_and_polars_input_give_equivalent_results(self):
        pandas_results, _ = self._run_qsar(pd.DataFrame(_make_bioactivity_data()))
        polars_results, _ = self._run_qsar(pl.DataFrame(_make_bioactivity_data()))
        pd.testing.assert_frame_equal(
            pandas_results.reset_index(drop=True), polars_results.reset_index(drop=True),
        )

    def test_regressor_pandas_and_polars_input_give_equivalent_results(self):
        model = DecisionTreeRegressor()
        pandas_results, _ = self._run_qsar(pd.DataFrame(_make_bioactivity_data()), model=model)
        polars_results, _ = self._run_qsar(pl.DataFrame(_make_bioactivity_data()), model=model)
        pd.testing.assert_frame_equal(
            pandas_results.reset_index(drop=True), polars_results.reset_index(drop=True),
        )


if __name__ == '__main__':
    unittest.main()
