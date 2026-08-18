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
from sklearn.base import ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.papyrus_scripts.modelling import (
    _fit_and_evaluate,
    _InsufficientData,
    crossvalidate_model,
    filter_molecular_descriptors,
    model_metrics,
    pcm,
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


class TestCrossvalidateModelScaling(unittest.TestCase):

    def test_scale_method_fit_only_on_fold_training_rows(self):
        # Regression test: scale_method used to be fit once on the whole
        # training set before folds.split() ran, so each fold's held-out
        # rows had already influenced the statistics used to scale them.
        class _RecordingScaler(StandardScaler):
            def __init__(self):
                super().__init__()
                self.fit_sizes = []

            def fit_transform(self, X, y=None, **kwargs):
                self.fit_sizes.append(len(X))
                return super().fit_transform(X, y, **kwargs)

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.random((20, 3)), columns=['a', 'b', 'c'])
        y = pd.Series(rng.random(20), name='y')
        data = pd.concat([y, X], axis=1)
        scaler = _RecordingScaler()
        kfold = KFold(n_splits=4, shuffle=True, random_state=0)

        crossvalidate_model(data, DecisionTreeRegressor(random_state=0), kfold, scale_method=scaler)

        # 4 folds of 15 training rows each (20 * 3/4), then one final fit
        # on the entire 20-row set for the "Full model".
        self.assertEqual(scaler.fit_sizes, [15, 15, 15, 15, 20])


class TestCrossvalidateModelVerbose(unittest.TestCase):

    def test_verbose_does_not_raise(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.random((20, 3)), columns=['a', 'b', 'c'])
        y = pd.Series(rng.random(20), name='y')
        data = pd.concat([y, X], axis=1)
        crossvalidate_model(data, DecisionTreeRegressor(random_state=0),
                            KFold(n_splits=4, shuffle=True, random_state=0), verbose=True)


class TestTrainTestProportionalGroupSplitVerbose(unittest.TestCase):

    def test_verbose_does_not_raise(self):
        data = pd.DataFrame({'x': range(20)})
        groups = [i % 5 for i in range(20)]
        train_test_proportional_group_split(data, groups, test_size=0.3, verbose=True)


class TestModelMetricsEdgeCases(unittest.TestCase):

    def test_unknown_model_type_raises(self):
        class _NotAModel:
            def predict(self, X):
                return np.zeros(len(X))
        with self.assertRaises(ValueError):
            model_metrics(_NotAModel(), np.array([0, 1]), np.zeros((2, 1)))

    def test_binary_classifier_single_proba_column(self):
        class _FakeClf(ClassifierMixin):
            classes_ = np.array([0, 1])

            def predict(self, X):
                return np.array([0, 1, 0, 1])

            def predict_proba(self, X):
                return np.array([[0.1], [0.9], [0.2], [0.8]])
        values = model_metrics(_FakeClf(), np.array([0, 1, 0, 1]), np.zeros((4, 1)))
        self.assertIn('AUC 1', values)

    def test_binary_classifier_degenerate_roc_auc_warns_and_sets_nan(self):
        # Modern sklearn only *warns* (UndefinedMetricWarning) for a
        # single-class y_true, it no longer raises ValueError - mock ROCAUC
        # itself to still exercise this defensive except branch.
        class _FakeClf(ClassifierMixin):
            classes_ = np.array([0, 1])

            def predict(self, X):
                return np.array([0, 0, 0, 0])

            def predict_proba(self, X):
                return np.array([[0.9, 0.1]] * 4)
        with patch('src.papyrus_scripts.modelling.ROCAUC', side_effect=ValueError('degenerate')):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                values = model_metrics(_FakeClf(), np.array([0, 0, 0, 0]), np.zeros((4, 1)))
        self.assertTrue(np.isnan(values['AUC 0']))

    def test_multiclass_degenerate_roc_auc_warns_and_sets_nan(self):
        class _FakeClf(ClassifierMixin):
            classes_ = np.array([0, 1, 2])

            def predict(self, X):
                return np.array([0, 0, 0, 0])

            def predict_proba(self, X):
                return np.array([[0.34, 0.33, 0.33]] * 4)
        with patch('src.papyrus_scripts.modelling.ROCAUC', side_effect=ValueError('degenerate')):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                values = model_metrics(_FakeClf(), np.array([0, 0, 0, 0]), np.zeros((4, 1)))
        self.assertTrue(np.isnan(values['AUC 1 vs 1']))
        self.assertTrue(np.isnan(values['AUC 1 vs All']))


class TestFitAndEvaluateSplitModes(unittest.TestCase):

    def _kwargs(self, data, **overrides):
        merge_on_values = pd.DataFrame({'id': range(len(data))}, index=data.index)
        base = dict(
            data=data, endpoint='y', model_type='regressor',
            model=DecisionTreeRegressor(random_state=0),
            merge_on='id', merge_on_values=merge_on_values,
            split_by='random', split_year=2015, test_set_size=0.3,
            cluster_method=None, custom_groups=None,
            features_to_ignore=['f1', 'f2'], drop_columns=['Year'],
            scale=False, scale_method=StandardScaler(),
            yscramble=False, stratify=False, folds=2, random_state=0,
            verbose=False, strict_split_checks=True,
        )
        base.update(overrides)
        return base

    def test_year_split_single_class_training_raises(self):
        n = 10
        data = pd.DataFrame({
            'y': [0] * n, 'Year': [2010] * 5 + [2020] * 5,
            'f1': range(n), 'f2': range(n),
        })
        with self.assertRaises(_InsufficientData):
            _fit_and_evaluate(**self._kwargs(data, model_type='classifier', split_by='year'))

    def test_year_split_single_class_test_raises_when_strict(self):
        n = 10
        data = pd.DataFrame({
            'y': [0, 1, 0, 1, 0] + [0] * 5,
            'Year': [2010] * 5 + [2020] * 5,
            'f1': range(n), 'f2': range(n),
        })
        with self.assertRaises(_InsufficientData):
            _fit_and_evaluate(**self._kwargs(data, model_type='classifier', split_by='year'))

    def test_cluster_split(self):
        from sklearn.cluster import KMeans
        n = 12
        data = pd.DataFrame({
            'y': range(1, n + 1), 'Year': [2010] * n,
            'f1': [0.0] * 6 + [10.0] * 6, 'f2': [0.0] * 6 + [10.0] * 6,
        })
        performance, _, _ = _fit_and_evaluate(**self._kwargs(
            data, split_by='cluster', cluster_method=KMeans(n_clusters=2, random_state=0, n_init=1),
        ))
        self.assertIn('Test set', performance.index)

    def test_custom_split(self):
        n = 10
        data = pd.DataFrame({'y': range(1, n + 1), 'Year': [2010] * n, 'f1': range(n), 'f2': range(n)})
        custom_groups = pd.DataFrame({'id': range(n), 'split': ['training'] * 7 + ['test'] * 3})
        performance, _, _ = _fit_and_evaluate(
            **self._kwargs(data, split_by='custom', custom_groups=custom_groups),
        )
        self.assertIn('Test set', performance.index)

    def test_custom_cluster_split(self):
        n = 10
        data = pd.DataFrame({'y': range(1, n + 1), 'Year': [2010] * n, 'f1': range(n), 'f2': range(n)})
        custom_groups = pd.DataFrame({'id': range(n), 'cluster': [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]})
        performance, _, _ = _fit_and_evaluate(
            **self._kwargs(data, split_by='custom-cluster', custom_groups=custom_groups),
        )
        self.assertIn('Test set', performance.index)

    def test_yscramble_does_not_raise(self):
        n = 20
        data = pd.DataFrame({
            'y': list(range(1, n + 1)), 'Year': [2010] * n,
            'f1': range(n), 'f2': range(n),
        })
        performance, _, _ = _fit_and_evaluate(**self._kwargs(data, yscramble=True))
        self.assertIn('Test set', performance.index)

    def test_stratified_classifier_scale_return_val(self):
        n = 20
        data = pd.DataFrame({
            'y': [0, 1] * 10, 'Year': [2010] * n,
            'f1': range(n), 'f2': range(n),
        })
        _, return_val, _ = _fit_and_evaluate(**self._kwargs(
            data, model_type='classifier', model=DecisionTreeClassifier(random_state=0),
            split_by='custom',
            custom_groups=pd.DataFrame({'id': range(n), 'split': ['training'] * 14 + ['test'] * 6}),
            stratify=True, scale=True,
        ))
        self.assertIn('scaler', return_val)
        self.assertIn('data_splitter', return_val)


class TestQsarPcmValidation(unittest.TestCase):

    def test_qsar_invalid_split_by_raises(self):
        with self.assertRaises(ValueError):
            qsar(pd.DataFrame(_make_bioactivity_data()), split_by='bogus')

    def test_qsar_cluster_split_without_cluster_method_raises(self):
        with self.assertRaises(ValueError):
            qsar(pd.DataFrame(_make_bioactivity_data()), split_by='cluster')

    def test_qsar_custom_split_without_custom_groups_raises(self):
        with self.assertRaises(ValueError):
            qsar(pd.DataFrame(_make_bioactivity_data()), split_by='custom')

    def test_qsar_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            qsar(pd.DataFrame(_make_bioactivity_data()), model=object())

    def test_qsar_no_targets_returns_empty_results(self):
        empty = pd.DataFrame(_make_bioactivity_data()).iloc[0:0]
        with patch(
            'src.papyrus_scripts.modelling.read_molecular_descriptors',
            return_value=pl.LazyFrame(_make_descriptors()),
        ):
            results, _ = qsar(empty, num_points=3, folds=2, split_year=2016)
        self.assertTrue(results.empty)

    def test_pcm_invalid_split_by_raises(self):
        with self.assertRaises(ValueError):
            pcm(pd.DataFrame(_make_bioactivity_data()), split_by='bogus')

    def test_pcm_cluster_split_without_cluster_method_raises(self):
        with self.assertRaises(ValueError):
            pcm(pd.DataFrame(_make_bioactivity_data()), split_by='cluster')

    def test_pcm_custom_split_without_custom_groups_raises(self):
        with self.assertRaises(ValueError):
            pcm(pd.DataFrame(_make_bioactivity_data()), split_by='custom')

    def test_pcm_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            pcm(pd.DataFrame(_make_bioactivity_data()), model=object())

    def test_pcm_too_few_datapoints_raises(self):
        data = pd.DataFrame(_make_bioactivity_data()).iloc[:2]
        with (
            patch(
                'src.papyrus_scripts.modelling.read_molecular_descriptors',
                return_value=pl.LazyFrame(_make_descriptors()),
            ),
            patch(
                'src.papyrus_scripts.modelling.read_protein_descriptors',
                return_value=pd.DataFrame({'target_id': ['T1'], 'Prot_1': [0.0]}),
            ),
        ):
            with self.assertRaises(ValueError):
                pcm(data, num_points=100)

    def test_pcm_insufficient_amplitude_raises(self):
        data = pd.DataFrame(_make_bioactivity_data())
        data['pchembl_value_Mean'] = 6.5  # no amplitude at all
        with (
            patch(
                'src.papyrus_scripts.modelling.read_molecular_descriptors',
                return_value=pl.LazyFrame(_make_descriptors()),
            ),
            patch(
                'src.papyrus_scripts.modelling.read_protein_descriptors',
                return_value=pd.DataFrame({'target_id': ['T1'], 'Prot_1': [0.0]}),
            ),
        ):
            with self.assertRaises(ValueError):
                pcm(data, num_points=3)

    def test_pcm_lazyframe_input(self):
        with (
            patch(
                'src.papyrus_scripts.modelling.read_molecular_descriptors',
                return_value=pl.LazyFrame(_make_descriptors()),
            ),
            patch(
                'src.papyrus_scripts.modelling.read_protein_descriptors',
                return_value=pd.DataFrame({'target_id': ['T1'], 'Prot_1': [0.0]}),
            ),
        ):
            performance, _ = pcm(pl.LazyFrame(_make_bioactivity_data()), num_points=3, folds=2, split_year=2016)
        self.assertIn('Test set', performance.index)

    def test_pcm_protein_descriptors_lazyframe(self):
        with (
            patch(
                'src.papyrus_scripts.modelling.read_molecular_descriptors',
                return_value=pl.LazyFrame(_make_descriptors()),
            ),
            patch(
                'src.papyrus_scripts.modelling.read_protein_descriptors',
                return_value=pl.LazyFrame({'target_id': ['T1'], 'Prot_1': [0.0]}),
            ),
        ):
            performance, _ = pcm(pd.DataFrame(_make_bioactivity_data()), num_points=3, folds=2, split_year=2016)
        self.assertIn('Test set', performance.index)

    def test_pcm_insufficient_data_inside_fit_and_evaluate_raises_valueerror(self):
        data = pd.DataFrame(_make_bioactivity_data())
        with (
            patch(
                'src.papyrus_scripts.modelling.read_molecular_descriptors',
                return_value=pl.LazyFrame(_make_descriptors()),
            ),
            patch(
                'src.papyrus_scripts.modelling.read_protein_descriptors',
                return_value=pd.DataFrame({'target_id': ['T1'], 'Prot_1': [0.0]}),
            ),
        ):
            # No row reaches Year 2030, so the temporal split's test set is
            # empty - _fit_and_evaluate raises _InsufficientData, which pcm()
            # re-raises as ValueError.
            with self.assertRaises(ValueError):
                pcm(data, num_points=3, split_year=2030)


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

    def test_polars_lazyframe_input_does_not_raise(self):
        results, _ = self._run_qsar(pl.LazyFrame(_make_bioactivity_data()))
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


def _make_multi_target_classifier_data():
    # T1: reaches a real fit. T2: too few points. T3: single activity class.
    # T4: minority class count (1) below folds (2). T5: passes qsar()'s own
    # checks but _fit_and_evaluate's temporal split finds no test-set data
    # (every row predates split_year) - covers the verbose branch on each
    # of these skip/error paths in one combined multi-target run.
    t1 = _make_bioactivity_data()
    extra = {
        'connectivity': ['C100', 'C101'] + ['C102', 'C103', 'C104'] + ['C105', 'C106', 'C107', 'C108']
                       + ['C109', 'C110', 'C111', 'C112'],
        'target_id': ['T2'] * 2 + ['T3'] * 3 + ['T4'] * 4 + ['T5'] * 4,
        'Activity_class': [None] * 13,
        'pchembl_value_Mean': [5.0, 5.2] + [5.0, 5.1, 5.2] + [5.0, 5.1, 5.2, 7.0] + [5.0, 5.1, 7.0, 7.1],
        'relation': ['='] * 13,
        'Year': [2010] * 13,
    }
    return {key: list(t1[key]) + extra[key] for key in t1}


def _make_multi_target_descriptors():
    connectivity = _make_descriptors()['connectivity'] + [
        'C100', 'C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109', 'C110', 'C111', 'C112',
    ]
    return {'connectivity': connectivity, 'Desc_1': [i / len(connectivity) for i in range(len(connectivity))]}


class TestQsarVerboseLoop(unittest.TestCase):

    def _run_qsar(self, data, model, **kwargs):
        with patch(
            'src.papyrus_scripts.modelling.read_molecular_descriptors',
            return_value=pl.LazyFrame(_make_multi_target_descriptors()),
        ):
            return qsar(data, model=model, num_points=3, folds=2, split_year=2016, verbose=True, **kwargs)

    def test_classifier_verbose_covers_every_skip_branch(self):
        results, _ = self._run_qsar(
            pd.DataFrame(_make_multi_target_classifier_data()), DecisionTreeClassifier(),
        )
        self.assertEqual(
            sorted(results.index.get_level_values('target').unique()), ['T1', 'T2', 'T3', 'T4', 'T5'],
        )

    def test_regressor_verbose_insufficient_amplitude(self):
        # Also a regression test: qsar() used to crash with KeyError when
        # every target was skipped/errored (no target ever reset_index()'d
        # a performance frame, so the final concat had no 'index' column
        # at all to set_index() on).
        n = 16
        data = _make_bioactivity_data()
        data['target_id'] = ['T2'] * n
        data['pchembl_value_Mean'] = [6.5] * n  # zero amplitude
        with patch(
            'src.papyrus_scripts.modelling.read_molecular_descriptors',
            return_value=pl.LazyFrame(_make_descriptors()),
        ):
            results, _ = qsar(pd.DataFrame(data), model=DecisionTreeRegressor(),
                             num_points=3, folds=2, split_year=2016, verbose=True)
        self.assertEqual(list(results.index.get_level_values('target').unique()), ['T2'])


if __name__ == '__main__':
    unittest.main()
