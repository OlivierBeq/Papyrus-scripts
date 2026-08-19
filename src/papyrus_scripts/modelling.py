# -*- coding: utf-8 -*-

"""Modelling capacities of the Papyrus-scripts from the Papyrus dataset."""

import warnings
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain, combinations
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost
from prodec.Descriptor import Descriptor
from prodec.Transform import Transform
from scipy.stats import kendalltau as kendallTau
from scipy.stats import pearsonr as pearsonR
from scipy.stats import spearmanr as spearmanR
from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
)
from sklearn.metrics import (
    explained_variance_score as eVar,
)
from sklearn.metrics import (
    matthews_corrcoef as MCC,
)
from sklearn.metrics import (
    max_error as maxE,
)
from sklearn.metrics import (
    mean_absolute_error as MAE,
)
from sklearn.metrics import (
    mean_gamma_deviance as MGD,
)
from sklearn.metrics import (
    mean_poisson_deviance as MPD,
)
from sklearn.metrics import (
    mean_squared_error as MSE,
)
from sklearn.metrics import (
    mean_squared_log_error as MSLE,
)
from sklearn.metrics import (
    r2_score as R2,
)
from sklearn.metrics import (
    roc_auc_score as ROCAUC,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

from .neuralnet import MultiTaskNNClassifier, MultiTaskNNRegressor, SingleTaskNNClassifier, SingleTaskNNRegressor
from .preprocess import yscrambling
from .reader import read_molecular_descriptors, read_protein_descriptors

#: Splitters whose .split() uses groups - passing it to any other splitter
#: (e.g. KFold) just triggers sklearn's ignored-groups warning.
_GROUP_AWARE_SPLITTERS = (
    GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, LeavePGroupsOut, StratifiedGroupKFold,
)


def filter_molecular_descriptors(data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
                                 column_name: str,
                                 keep_values: Iterable,
                                 progress: bool = True,
                                 total: int | None = None) -> pd.DataFrame:
    """Filter the data so that the desired column contains only the desired data.

    :param data: data to be filtered - pandas/polars DataFrame or polars
        LazyFrame
    :param column_name: name of the column to apply the filter on
    :param keep_values: allowed values
    :param progress: unused - kept for API stability
    :param total: unused - kept for API stability
    :return: a pandas dataframe
    """
    if isinstance(data, pd.DataFrame):
        return data[data[column_name].isin(keep_values)]
    # Filter before collecting a LazyFrame so polars pushes the predicate down.
    filtered = data.filter(pl.col(column_name).is_in(list(keep_values)))
    if isinstance(filtered, pl.LazyFrame):
        filtered = filtered.collect()
    return filtered.to_pandas()


def model_metrics(model: RegressorMixin | ClassifierMixin,
                  y_true: pd.Series | np.ndarray,
                  x_test: pd.DataFrame) -> dict[str, Any]:
    """Determine performance metrics of a model.

    Beware R2 = 1 - (Residual sum of squares) / (Total sum of squares) != (Pearson r)²

    R2_0, R2_0_prime, K and k_prime are derived from
    Tropsha, A., & Golbraikh, A. (2010).
    Predictive Quantitative Structure–Activity Relationships Modeling.
    In J.-L. Faulon & A. Bender (Eds.),
    Handbook of Chemoinformatics Algorithms.
    Chapman and Hall/CRC.
    https://www.taylorfrancis.com/books/9781420082999

    :param model: model to check the performance of
    :param y_true: true labels
    :param x_test: testing set of features
    :return: a dictionary of metrics
    """
    y_pred = model.predict(x_test)
    # Regression metrics
    if isinstance(model, (RegressorMixin, SingleTaskNNRegressor, MultiTaskNNRegressor)):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # Slope of predicted vs observed
        k = np.dot(y_true, y_pred) / np.sum(y_true ** 2)
        # Slope of observed vs predicted
        k_prime = np.dot(y_true, y_pred) / np.sum(y_pred ** 2)
        # Mean averages
        y_true_mean = y_true.mean()
        y_pred_mean = y_pred.mean()
        # Pearson/Spearman r and R2_0/R'2_0 are undefined (and warn) when
        # y_true or y_pred is constant, e.g. a degenerate tiny test fold.
        has_variance = len(y_pred) >= 2 and y_true.std() > 0 and y_pred.std() > 0
        return {'number': y_true.size,
                'R2': R2(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'MSE': MSE(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'RMSE': MSE(y_true, y_pred) ** 0.5 if len(y_pred) >= 2 else 0,
                'MSLE': MSLE(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'RMSLE': np.sqrt(MSLE(y_true, y_pred)) if len(y_pred) >= 2 else 0,
                'MAE': MAE(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Explained Variance': eVar(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Max Error': maxE(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Mean Poisson Distrib': MPD(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Mean Gamma Distrib': MGD(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Pearson r': pearsonR(y_true, y_pred)[0] if has_variance else 0,
                'Spearman r': spearmanR(y_true, y_pred)[0] if has_variance else 0,
                'Kendall tau': kendallTau(y_true, y_pred)[0] if len(y_pred) >= 2 else 0,
                'R2_0 (pred. vs. obs.)': 1 - (np.sum((y_true - k_prime * y_pred) ** 2) /
                                              np.sum((y_true - y_true_mean) ** 2)) if has_variance else 0,
                'R\'2_0 (obs. vs. pred.)': 1 - (np.sum((y_pred - k * y_true) ** 2) /
                                                np.sum((y_pred - y_pred_mean) ** 2)) if has_variance else 0,
                'k slope (pred. vs obs.)': k,
                'k\' slope (obs. vs pred.)': k_prime,
                }
    # Classification
    elif isinstance(model, (ClassifierMixin, SingleTaskNNClassifier, MultiTaskNNClassifier)):
        # ROC AUC is undefined when y_true holds a single class; sklearn now
        # warns rather than raising, so check upfront instead of catching it.
        single_class = len(set(y_true)) < 2
        if single_class:
            warnings.warn('Only one class present in y_true. ROC AUC score is not defined in that case. '
                          'Stratify your folds to avoid such warning.', stacklevel=2)
        # Binary classification
        if len(model.classes_) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=model.classes_).ravel()
            values = {}
            values['MCC'] = MCC(y_true, y_pred)
            values[':'.join(str(x) for x in model.classes_)] = ':'.join([str(int(sum(y_true == class_)))
                                                                         for class_ in model.classes_])
            values['ACC'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
            values['BACC'] = (tp / (tp + fn) + tn / (tn + fp)) / 2
            values['Sensitivity'] = tp / (tp + fn) if tp + fn != 0 else 0
            values['Specificity'] = tn / (tn + fp) if tn + fp != 0 else 0
            values['PPV'] = tp / (tp + fp) if tp + fp != 0 else 0
            values['NPV'] = tn / (tn + fn) if tn + fn != 0 else 0
            values['F1'] = 2 * values['Sensitivity'] * values['PPV'] / (values['Sensitivity'] + values['PPV']) \
                if (values['Sensitivity'] + values['PPV']) != 0 \
                else 0
            if hasattr(model, "predict_proba"): # able to predict probability
                y_probas = model.predict_proba(x_test)
                if y_probas.shape[1] == 1:
                    y_proba = y_probas.ravel()
                    values['AUC 1'] = np.nan if single_class else ROCAUC(y_true, y_proba)
                else:
                    for i in range(len(model.classes_)):
                        y_proba = y_probas[:, i].ravel()
                        values[f'AUC {model.classes_[i]}'] = np.nan if single_class else ROCAUC(y_true, y_proba)
        # Multiclasses
        else:
            i = 0
            values = {}
            for contingency_matrix in multilabel_confusion_matrix(y_true, y_pred):
                tn, fp, fn, tp = contingency_matrix.ravel()
                values[f'{model.classes_[i]}|MCC'] = MCC(y_true, y_pred)
                values[f'{model.classes_[i]}|number'] = int(sum(y_true == model.classes_[i]))
                values[f'{model.classes_[i]}|ACC'] = (tp + tn) / (tp + tn + fp + fn) \
                    if (tp + tn + fp + fn) != 0\
                    else 0
                values[f'{model.classes_[i]}|BACC'] = (tp / (tp + fn) + tn / (tn + fp)) / 2
                values[f'{model.classes_[i]}|Sensitivity'] = tp / (tp + fn) if tp + fn != 0 else 0
                values[f'{model.classes_[i]}|Specificity'] = tn / (tn + fp) if tn + fp != 0 else 0
                values[f'{model.classes_[i]}|PPV'] = tp / (tp + fp) if tp + fp != 0 else 0
                values[f'{model.classes_[i]}|NPV'] = tn / (tn + fn) if tn + fn != 0 else 0
                values[f'{model.classes_[i]}|F1'] = \
                    2 * values[f'{model.classes_[i]}|Sensitivity'] * values[f'{model.classes_[i]}|PPV'] / \
                    (values[f'{model.classes_[i]}|Sensitivity'] + values[f'{model.classes_[i]}|PPV']) \
                     if (values[f'{model.classes_[i]}|Sensitivity'] + values[f'{model.classes_[i]}|PPV']) != 0 \
                     else 0
                i += 1
            if hasattr(model, "predict_proba"): # able to predict probability
                y_probas = model.predict_proba(x_test)
                if single_class:
                    values['AUC 1 vs 1'] = np.nan
                    values['AUC 1 vs All'] = np.nan
                else:
                    values['AUC 1 vs 1'] = ROCAUC(y_true, y_probas, average="macro", multi_class="ovo")
                    values['AUC 1 vs All'] = ROCAUC(y_true, y_probas, average="macro", multi_class="ovr")
        return values
    else:
        raise ValueError('model can only be classifier or regressor.')


def crossvalidate_model(data: pd.DataFrame,
                        model: RegressorMixin | ClassifierMixin,
                        folds: BaseCrossValidator,
                        groups: list[int] | pd.Series | None = None,
                        scale_method: TransformerMixin | None = None,
                        verbose: bool = False,
                        leave: bool = True,
                        ) -> tuple[pd.DataFrame, dict[str, RegressorMixin | ClassifierMixin]]:
    """Create a machine learning model predicting values in the first column.

    :param data: data containing the dependent vairable (in the first column) and other features
    :param model: estimator (either classifier or regressor) to use for model building
    :param folds: cross-validator
    :param groups: groups to split the labels according to
    :param scale_method: if given, fit anew on each fold's training split
       only (never on its held-out split) to avoid leaking test-fold
       statistics into the scaling; also fit once on the entire dataset for
       the final "Full model" - left fitted on that full-dataset call when
       this function returns, so a caller can reuse it as-is
    :param verbose: whether to show fold progression
    :param leave: whether this function's progress bar stays on screen once done
    :return: cross-validated performance and model trained on the entire dataset
    """
    X, y = data.iloc[:, 1:], data.iloc[:, 0].values.ravel()
    fold_metrics: list[dict[str, Any]] = []
    pbar = tqdm(desc='Fitting model', total=folds.n_splits + 1, leave=leave) if verbose else None
    try:
        models: dict[str, RegressorMixin | ClassifierMixin] = {}
        split_groups = groups if isinstance(folds, _GROUP_AWARE_SPLITTERS) else None
        # Perform cross-validation
        for i, (train, test) in enumerate(folds.split(X, y, split_groups)):
            if verbose:
                pbar.set_description(f'Fitting model on fold {i + 1}', refresh=True)
            X_train, X_test = X.iloc[train, :], X.iloc[test, :]
            if scale_method is not None:
                X_train = pd.DataFrame(scale_method.fit_transform(X_train),
                                       index=X_train.index, columns=X_train.columns)
                X_test = pd.DataFrame(scale_method.transform(X_test),
                                      index=X_test.index, columns=X_test.columns)
            model.fit(X_train, y[train])
            models[f'Fold {i + 1}'] = deepcopy(model)
            fold_metrics.append(model_metrics(model, y[test], X_test))
            if verbose:
                pbar.update()
        # Organize result in a dataframe
        performance = pd.DataFrame(fold_metrics)
        performance.index = [f'Fold {i + 1}' for i in range(folds.n_splits)]
        # Add average and sd of performance
        performance.loc['Mean'] = [np.mean(performance[col]) if ':' not in col else '-' for col in performance]
        performance.loc['SD'] = [np.std(performance[col]) if ':' not in col else '-' for col in performance]
        # Fit model on the entire dataset
        if verbose:
            pbar.set_description('Fitting model on entire training set', refresh=True)
        if scale_method is not None:
            X = pd.DataFrame(scale_method.fit_transform(X), index=X.index, columns=X.columns)
        model.fit(X, y)
        models['Full model'] = deepcopy(model)
        if verbose:
            pbar.update()
    finally:
        if pbar is not None:
            # avoids tqdm.notebook leaving a stuck widget if closed early
            pbar.n = pbar.total
            pbar.close()
    return performance, models


def train_test_proportional_group_split(data: pd.DataFrame,
                                        groups: list[int] | np.ndarray,
                                        test_size: float = 0.30,
                                        verbose: bool = False,
                                        ) -> tuple[pd.DataFrame, pd.DataFrame, list[int], tuple[int, ...]]:
    """Split the data into training and test sets according to the groups that respect most test_size (based on MSE).

    :param data: the data to be split up into training and test sets
    :param groups: groups to split the data according to
    :param test_size: approximate proportion of the input dataset to determine the test set
    :param verbose: whether to log to stdout or not
    :return: training and test sets and training and test groups
    """
    counts = Counter(groups)
    size = sum(counts.values())
    # Get ordered permutations of groups without repetitions
    permutations = list(chain.from_iterable(combinations(counts.keys(), r) for r in range(len(counts))))
    # Get proportion of each permutation
    proportions = [sum(counts[x] for x in p) / size for p in permutations]
    # Get permutation minimizing difference to test_size
    best, proportion = min(zip(permutations, proportions, strict=True), key=lambda x: (x[1] - test_size) ** 2)
    del counts, permutations, proportions
    if verbose:
        print(f'Best group permutation corresponds to {proportion:.2%} of the data')
    # Get test set assignment
    assignment = np.array([group in best for group in groups])
    opposite = np.logical_not(assignment)
    # Get training groups
    t_groups = [x for x in groups if x not in best]
    return data[opposite], data[assignment], t_groups, best


def _leave_bar(leave_level: int, depth: int) -> bool:
    """Whether a bar nested *depth* levels deep (1 = outermost) leaves, per ``leave_level``."""
    return leave_level < 0 or leave_level >= depth


class _InsufficientDataError(Exception):
    """Raised by _fit_and_evaluate when a split/class-balance check fails.

    Caught and logged by qsar() (per target, loop continues), re-raised as
    a ValueError by pcm() (single combined model, nothing to continue to).
    """


def _fit_and_evaluate(data: pd.DataFrame,
                      endpoint: str,
                      model_type: str,
                      model: RegressorMixin | ClassifierMixin,
                      merge_on: str,
                      merge_on_values: pd.DataFrame,
                      split_by: str,
                      split_year: int,
                      test_set_size: float,
                      cluster_method: ClusterMixin | None,
                      custom_groups: pd.DataFrame | None,
                      features_to_ignore: list[str],
                      drop_columns: list[str],
                      scale: bool,
                      scale_method: TransformerMixin,
                      yscramble: bool,
                      stratify: bool,
                      folds: int,
                      random_state: int,
                      verbose: bool,
                      strict_split_checks: bool,
                      cv_leave: bool = True,
                      ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, RegressorMixin | ClassifierMixin]]:
    """Split *data*, fit *model* via cross-validation, and evaluate on the held-out test set.

    Shared by qsar() (once per target) and pcm() (once on the whole dataset).

    :param drop_columns: columns to drop before splitting into X/y - qsar()
        also drops 'target_id' here (loops per target); pcm() already
        dropped it after merging protein descriptors
    :param strict_split_checks: qsar() also requires >= *folds* training
        rows and balanced test-set classes; pcm() checks neither
    :param cv_leave: passed through as crossvalidate_model()'s ``leave``
    :raises _InsufficientDataError: if a split/class-balance check fails
    :returns: (performance, return_val, cv_models) - return_val holds the
        scaler/label_encoder/data_splitter (as applicable); cv_models is
        crossvalidate_model's per-fold-plus-"Full model" dict
    """
    if split_by.lower() == 'year':
        test_set = data[data['Year'] >= split_year]
        if test_set.empty:
            raise _InsufficientDataError(f'No test data for temporal split at {split_year}')
        training_set = data[~data.index.isin(test_set.index)]
        if training_set.empty or (strict_split_checks and training_set.shape[0] < folds):
            raise _InsufficientDataError(f'Not enough training data for temporal split at {split_year}')
        if model_type == 'classifier':
            train_data_classes = Counter(training_set[endpoint])
            if len(train_data_classes) < 2:
                raise _InsufficientDataError(
                    f'Only one activity class in the training set for temporal split at {split_year}')
            if strict_split_checks:
                test_data_classes = Counter(test_set[endpoint])
                if len(test_data_classes) < 2:
                    raise _InsufficientDataError(
                        f'Only one activity class in the test set for temporal split at {split_year}')
        training_groups = training_set['Year']
    elif split_by.lower() == 'random':
        training_groups = None
        training_set, test_set = train_test_split(data, test_size=test_set_size, random_state=random_state)
    elif split_by.lower() == 'cluster':
        if cluster_method is None:
            raise RuntimeError('cluster_method missing despite qsar()/pcm() validating it upfront')
        groups = cluster_method.fit_predict(data.drop(columns=features_to_ignore))
        training_set, test_set, training_groups, _ = train_test_proportional_group_split(data, groups,
                                                                                         test_set_size,
                                                                                         verbose=verbose)
    elif split_by.lower() == 'custom-cluster':
        # Merge from custom split DataFrame
        groups = merge_on_values.merge(custom_groups, on=merge_on).iloc[:, 1].tolist()
        training_set, test_set, training_groups, _ = train_test_proportional_group_split(data, groups,
                                                                                         test_set_size,
                                                                                         verbose=verbose)
    elif split_by.lower() == 'custom':
        # Merge from custom split DataFrame
        groups = merge_on_values.merge(custom_groups, on=merge_on)
        training_set = data[merge_on_values.squeeze().isin(groups[groups.iloc[:, 1] == 'training'][merge_on])]
        test_set = data[merge_on_values.squeeze().isin(groups[groups.iloc[:, 1] == 'test'][merge_on])]
        training_groups = None
    # Drop columns not used for training
    training_set = training_set.drop(columns=drop_columns)
    test_set = test_set.drop(columns=drop_columns)
    X_train, y_train = training_set.drop(columns=[endpoint]), training_set.loc[:, endpoint]
    X_test, y_test = test_set.drop(columns=[endpoint]), test_set.loc[:, endpoint]
    # Scaling itself happens inside crossvalidate_model, fold-by-fold, to
    # avoid leaking a fold's held-out rows into its own scaling statistics.
    # Encode labels
    lblenc = None
    if model_type == 'classifier':
        lblenc = LabelEncoder()
        y_train = pd.Series(data=lblenc.fit_transform(y_train),
                            index=y_train.index, dtype=y_train.dtype,
                            name=y_train.name)
        y_test = pd.Series(data=lblenc.transform(y_test),
                           index=y_test.index, dtype=y_test.dtype,
                           name=y_test.name)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
    # Reorganize data
    training_set = pd.concat([y_train, X_train], axis=1)
    test_set = pd.concat([y_test, X_test], axis=1)
    del X_train, y_train, X_test, y_test
    # Y-scrambling - yscrambling() is polars-only, so round-trip through it
    # and restore the pandas index it doesn't carry.
    if yscramble:
        training_index, test_index = training_set.index, test_set.index
        training_set = yscrambling(pl.from_pandas(training_set), y_var=endpoint,
                                   random_state=random_state).to_pandas()
        training_set.index = training_index
        test_set = yscrambling(pl.from_pandas(test_set), y_var=endpoint,
                               random_state=random_state).to_pandas()
        test_set.index = test_index
    # Make sure enough data
    if model_type == 'classifier':
        train_data_classes = Counter(training_set[endpoint])
        if not np.all(np.array(list(train_data_classes.values())) > folds):
            raise _InsufficientDataError(
                f'Not enough data in minority class of the training set for all {folds} folds')
        if strict_split_checks:
            test_data_classes = Counter(test_set[endpoint])
            if not np.all(np.array(list(test_data_classes.values())) > folds):
                raise _InsufficientDataError(
                    f'Not enough data in minority class of the test set for all {folds} folds')
    # Define folding scheme for cross validation
    if stratify and model_type == 'classifier':
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    performance, cv_models = crossvalidate_model(
        training_set, model, kfold, training_groups,
        scale_method=scale_method if scale else None, verbose=verbose, leave=cv_leave,
    )
    full_model = cv_models['Full model']
    X_test, y_test = test_set.iloc[:, 1:], test_set.iloc[:, 0].values.ravel()
    if scale:
        # scale_method was last fit inside crossvalidate_model on the whole
        # training set (for the "Full model") - reuse that fit here so
        # X_test is scaled consistently with full_model.
        X_test = pd.DataFrame(scale_method.transform(X_test), index=X_test.index, columns=X_test.columns)
    performance.loc['Test set'] = model_metrics(full_model, y_test, X_test)
    # Formatting return values
    return_val: dict[str, Any] = {}
    if scale:
        return_val['scaler'] = deepcopy(scale_method)
    if model_type == 'classifier':
        return_val['label_encoder'] = deepcopy(lblenc)
        if stratify:
            return_val['data_splitter'] = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        return_val['data_splitter'] = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    return performance, return_val, cv_models


def qsar(data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
         endpoint: str = 'pchembl_value_Mean',
         num_points: int = 30,
         delta_activity: float = 2,
         version: str = 'latest',
         descriptors: str = 'mold2',
         descriptor_path: str | None = None,
         descriptor_chunksize: int | None = 50000,
         activity_threshold: float = 6.5,
         model: RegressorMixin | ClassifierMixin | None = None,
         folds: int = 5,
         stratify: bool = False,
         split_by: str = 'Year',
         split_year: int = 2013,
         test_set_size: float = 0.30,
         cluster_method: ClusterMixin | None = None,
         custom_groups: pd.DataFrame | None = None,
         scale: bool = False,
         scale_method: TransformerMixin | None = None,
         yscramble: bool = False,
         random_state: int = 1234,
         verbose: bool = True,
         leave_level: int = -1,
         ) -> tuple[pd.DataFrame, dict[str,
                                       None | (TransformerMixin | LabelEncoder |
                                                      BaseCrossValidator | dict[str, ClassifierMixin])]]:
    """Create QSAR models for as many targets as meet the given thresholds.

    Targets are modelled only if they meet the selected data source(s),
    data quality, minimum number of datapoints and minimum activity
    amplitude requirements.

    :param data: Papyrus activity data; a ``pl.DataFrame``/``pl.LazyFrame``
        is materialised into a pandas DataFrame immediately (not lazily)
    :param endpoint: value to be predicted or to derive classes from
    :param num_points: minimum number of points for the activity of a target to be modelled
    :param delta_activity: minimum difference between most and least active compounds for a target to be modelled
    :param version: version of the papyrus dataset to use for modelling
    :param descriptors: type of desriptors to be used for model training
    :param descriptor_path: path to Papyrus descriptors (default: pystow's default path)
    :param descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded (None disables chunking)
    :param activity_threshold: threshold activity between acvtive and inactive compounds (ignored if using a regressor)
    :param model: machine learning model to be used for QSAR modelling
    :param folds: number of cross-validation folds to be performed
    :param stratify: whether to stratify folds for cross validation, ignored if model is RegressorMixin
    :param split_by: how should folds be determined {'random', 'Year', 'cluster', 'custom-cluster' 'custom'}
    If 'random', exactly test_set_size is extracted for test set.
    If 'Year', the size of the test and training set are not looked at
    If 'cluster', 'custom-cluster', the groups giving proportion closest to test_set_size will be used to
    define the test set. 'cluster' uses `cluster_method` to define groups while 'custom-cluster' uses user provided
    groups and creates the best suited proportional split among them.
    If 'custom', the groups to be used untouched, specifying either 'training' or 'test' for each entry (other labels
    are disregarded).
    :param split_year: Year from which on the test set is extracted (ignored if split_by is not 'Year')
    :param test_set_size: proportion of the dataset to be used as test set
    :param cluster_method: clustering method to use to extract test set and cross-validation folds
    (ignored if split_by is not 'cluster')
    :param custom_groups: custom groups to use to extract test set and cross-validation fold
    (ignored if split_by is not 'custom-cluster' or 'custom').
    Groups must be a pandas DataFrame with only two Series.The first Series is either InChIKey or connectivity
    (depending on whether stereochemistry data are being use or not). The second Series must be the group assignment
    of each compound specifying either 'training' or 'test' for each entry (other labels are disregarded) when
    `split_by` is 'custom' or cluster membership when `split_by` is 'custom-cluster'.
    :param scale: should the features be scaled using the custom scaling_method
    :param scale_method: scaling method to be applied to features (ignored if scale is False)
    :param yscramble: should the endpoint be shuffled to compare performance to the unshuffled endpoint
    :param random_state: seed to use for train/test splitting and KFold shuffling
    :param verbose: log details to stdout
    :param leave_level: how many nested progress bar levels stay on screen
        once done (level 1: per-target; level 2: per-fold); ``0`` none,
        ``-1`` all (default)
    :return: both:
    - a dataframe of the cross-validation results where each line is a fold of QSAR modelling of an accession
    - a dictionary of the feature scaler (if used), label encoder (if mode is a classifier),
    the data splitter for cross-validation, and for each accession in the data:
    the fitted models on each cross-validation fold and the model fitted on the complete training set.
    """
    if model is None:
        model = xgboost.XGBRegressor(verbosity=0)
    if scale_method is None:
        scale_method = StandardScaler()
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    if split_by.lower() not in ['year', 'random', 'cluster', 'custom-cluster', 'custom']:
        raise ValueError("split not supported, must be one of {'Year', 'random', 'cluster',"
                         "'custom-cluster', 'custom'}")
    if split_by.lower() == 'cluster' and cluster_method is None:
        raise ValueError("cluster_method must be given if split_by is 'cluster'")
    if split_by.lower() in ('custom-cluster', 'custom') and custom_groups is None:
        raise ValueError("custom_groups must be given if split_by is 'custom-cluster' or 'custom'")
    if not isinstance(model, (RegressorMixin, ClassifierMixin)):
        raise ValueError('model type can only be a Scikit-Learn compliant regressor or classifier')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if isinstance(model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
        warnings.filterwarnings("ignore", category=UserWarning)
    model_type = 'regressor' if isinstance(model, RegressorMixin) else 'classifier'
    # Keep only required fields
    merge_on = 'connectivity' if 'connectivity' in data.columns else 'InChIKey'
    if model_type == 'regressor':
        features_to_ignore = [merge_on, 'target_id', endpoint, 'Year']
        data = data[data['relation'] == '='][features_to_ignore]
    else:
        features_to_ignore = [merge_on, 'target_id', 'Activity_class', 'Year']
        preserved = data[~data['Activity_class'].isna()]
        preserved = preserved.drop(
            columns=[col for col in preserved if col not in [merge_on, 'target_id', 'Activity_class', 'Year']])
        active = data[data['Activity_class'].isna() & (data[endpoint] > activity_threshold)]
        active = active[~active['relation'].str.contains('<')][features_to_ignore].copy()
        active.loc[:, 'Activity_class'] = 'A'
        inactive = data[data['Activity_class'].isna() & (data[endpoint] <= activity_threshold)]
        inactive = inactive[~inactive['relation'].str.contains('>')][features_to_ignore].copy()
        inactive.loc[:, 'Activity_class'] = 'N'
        data = pd.concat([preserved, active, inactive])
        # Change endpoint
        endpoint = 'Activity_class'
        del preserved, active, inactive
    # Get and merge molecular descriptors
    descs = read_molecular_descriptors(descriptors, 'connectivity' not in data.columns,
                                       version, descriptor_chunksize, descriptor_path)
    descs = filter_molecular_descriptors(descs, merge_on, data[merge_on].unique())
    data = data.merge(descs, on=merge_on)
    merge_on_values = data[[merge_on]]
    data = data.drop(columns=[merge_on])
    del descs
    # Table of results
    fold_results: list[pd.DataFrame] = []
    models: dict[str, dict[str, RegressorMixin | ClassifierMixin] | None] = {}
    # Overwritten by each successfully-fitted target - the last success's
    # value is what ends up in this function's returned return_val.
    final_return_val: dict[str, Any] = {}
    targets = list(data['target_id'].unique())
    n_targets = len(targets)
    cv_leave = _leave_bar(leave_level, 2)
    pbar = tqdm(total=n_targets, smoothing=0.1, leave=_leave_bar(leave_level, 1)) if verbose else None
    try:
        # Build QSAR model for targets reaching criteria
        for i_target in range(n_targets - 1, -1, -1):
            tmp_data = data[data['target_id'] == targets[i_target]]
            tmp_merge_on_values = merge_on_values[merge_on_values.index.isin(tmp_data.index)]
            if verbose:
                pbar.set_description(
                    f'Building QSAR for target: {targets[i_target]} #datapoints {tmp_data.shape[0]}',
                    refresh=True)
            # Insufficient data points
            if tmp_data.shape[0] < num_points:
                if model_type == 'regressor':
                    fold_results.append(pd.DataFrame([[targets[i_target],
                                                  tmp_data.shape[0],
                                                  f'Number of points {tmp_data.shape[0]} < {num_points}']],
                                                columns=['target', 'number', 'error']))
                else:
                    data_classes = Counter(tmp_data[endpoint])
                    fold_results.append(
                        pd.DataFrame([[targets[i_target],
                                       ':'.join(str(data_classes.get(x, 0)) for x in ['A', 'N']),
                                       f'Number of points {tmp_data.shape[0]} < {num_points}']],
                                     columns=['target', 'A:N', 'error']))
                if verbose:
                    pbar.update()
                models[targets[i_target]] = None
                continue
            if model_type == 'regressor':
                min_activity = tmp_data[endpoint].min()
                max_activity = tmp_data[endpoint].max()
                delta = max_activity - min_activity
                # Not enough activity amplitude
                if delta < delta_activity:
                    fold_results.append(pd.DataFrame([[targets[i_target],
                                                  tmp_data.shape[0],
                                                  f'Delta activity {delta} < {delta_activity}']],
                                                columns=['target', 'number', 'error']))
                    if verbose:
                        pbar.update()
                    models[targets[i_target]] = None
                    continue
            else:
                data_classes = Counter(tmp_data[endpoint])
                # Only one activity class
                if len(data_classes) == 1:
                    fold_results.append(
                        pd.DataFrame([[targets[i_target],
                                       ':'.join(str(data_classes.get(x, 0)) for x in ['A', 'N']),
                                       'Only one activity class']],
                                     columns=['target', 'A:N', 'error']))
                    if verbose:
                        pbar.update()
                    models[targets[i_target]] = None
                    continue
                # Not enough data in minority class for all folds
                elif not all(x >= folds for x in data_classes.values()):
                    fold_results.append(
                        pd.DataFrame([[targets[i_target],
                                       ':'.join(str(data_classes.get(x, 0)) for x in ['A', 'N']),
                                       f'Not enough data in minority class for all {folds} folds']],
                                     columns=['target', 'A:N', 'error']))
                    if verbose:
                        pbar.update()
                    models[targets[i_target]] = None
                    continue
            # Split, fit and evaluate this target - see _fit_and_evaluate's
            # docstring for the pipeline shared with pcm().
            try:
                performance, final_return_val, cv_models = _fit_and_evaluate(
                    tmp_data, endpoint, model_type, model, merge_on, tmp_merge_on_values,
                    split_by, split_year, test_set_size, cluster_method, custom_groups,
                    features_to_ignore, ['Year', 'target_id'], scale, scale_method,
                    yscramble, stratify, folds, random_state, verbose,
                    strict_split_checks=True, cv_leave=cv_leave,
                )
            except _InsufficientDataError as exc:
                if model_type == 'regressor':
                    fold_results.append(pd.DataFrame([[targets[i_target], tmp_data.shape[0], str(exc)]],
                                                columns=['target', 'number', 'error']))
                else:
                    data_classes = Counter(tmp_data[endpoint])
                    fold_results.append(pd.DataFrame([[targets[i_target],
                                                  ':'.join(str(data_classes.get(x, 0)) for x in ['A', 'N']),
                                                  str(exc)]],
                                                columns=['target', 'A:N', 'error']))
                if verbose:
                    pbar.update()
                models[targets[i_target]] = None
                continue
            performance.loc[:, 'target'] = targets[i_target]
            fold_results.append(performance.reset_index())
            models[targets[i_target]] = cv_models
            if verbose:
                pbar.update()
    finally:
        if pbar is not None:
            pbar.n = pbar.total
            pbar.close()
    if isinstance(model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
        warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    return_val = {**final_return_val, **models}
    if not fold_results:
        return pd.DataFrame(), return_val
    results = pd.concat(fold_results, axis=0)
    if 'index' not in results.columns:
        # No target reached a real fit (every one was skipped/errored) - the
        # 'index' column normally comes from performance.reset_index() on a
        # successful fit, so it doesn't exist at all in that case.
        results['index'] = 0
    results = results.set_index(['target', 'index'])
    results.index.names = ['target', None]
    return results, return_val


def pcm(data: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
        endpoint: str = 'pchembl_value_Mean',
        num_points: int = 30,
        delta_activity: float = 2,
        version: str = 'latest',
        mol_descriptors: str = 'mold2',
        mol_descriptor_path: str | None = None,
        mol_descriptor_chunksize: int | None = 50000,
        prot_sequences_path: str | None = None,
        prot_descriptors: str | Descriptor | Transform = 'unirep',
        prot_descriptor_path: str | None = None,
        prot_descriptor_chunksize: int | None = 50000,
        activity_threshold: float = 6.5,
        model: RegressorMixin | ClassifierMixin | None = None,
        folds: int = 5,
        stratify: bool = False,
        split_by: str = 'Year',
        split_year: int = 2013,
        test_set_size: float = 0.30,
        cluster_method: ClusterMixin | None = None,
        custom_groups: pd.DataFrame | None = None,
        scale: bool = False,
        scale_method: TransformerMixin | None = None,
        yscramble: bool = False,
        random_state: int = 1234,
        verbose: bool = True,
        leave_level: int = -1,
        ) -> tuple[pd.DataFrame, dict[str,
                                      (TransformerMixin | LabelEncoder |
                                            BaseCrossValidator | RegressorMixin | ClassifierMixin)]]:
    """Create a single PCM model covering all targets that meet the given thresholds.

    Data is filtered to the selected data source(s), data quality, minimum
    number of datapoints and minimum activity amplitude requirements before
    fitting.

    :param data: Papyrus activity data; a ``pl.DataFrame``/``pl.LazyFrame``
        is materialised into a pandas DataFrame immediately (not lazily)
    :param endpoint: value to be predicted or to derive classes from
    :param num_points: minimum number of points for the activity of a target to be modelled
    :param delta_activity: minimum difference between most and least active compounds for a target to be modelled
    :param version: version of the papyrus dataset to use for modelling
    :param mol_descriptors: type of desriptors to be used for model training
    :param mol_descriptor_path: path to Papyrus descriptors
    :param mol_descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded
    (None disables chunking)
    :param prot_sequences_path: path to Papyrus sequences
    :param prot_descriptors: type of desriptors to be used for model training
    :param prot_descriptor_path: path to Papyrus descriptors
    :param prot_descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded
    (None disables chunking)
    :param activity_threshold: threshold activity between acvtive and inactive compounds (ignored if using a regressor)
    :param model: machine learning model to be used for PCM modelling
    :param folds: number of cross-validation folds to be performed
    :param stratify: whether to stratify folds for cross validation, ignored if model is RegressorMixin
    :param split_by: how should folds be determined {'random', 'Year', 'cluster', 'custom-cluster' 'custom'}
    If 'random', exactly test_set_size is extracted for test set.
    If 'Year', the size of the test and training set are not looked at
    If 'cluster', 'custom-cluster', the groups giving proportion closest to test_set_size will be used to
    define the test set. 'cluster' uses `cluster_method` to define groups while 'custom-cluster' uses user provided
    groups and creates the best suited proportional split among them.
    If 'custom', the groups to be used untouched, specifying either 'training' or 'test' for each entry (other labels
    are disregarded).
    :param split_year: Year from which on the test set is extracted (ignored if split_by is not 'Year')
    :param test_set_size: proportion of the dataset to be used as test set
    :param cluster_method: clustering method to use to extract test set and cross-validation folds
    (ignored if split_by is not 'cluster')
    :param custom_groups: custom groups to use to extract test set and cross-validation fold
    (ignored if split_by is not 'custom-cluster' or 'custom').
    Groups must be a pandas DataFrame with only two Series.The first Series is either InChIKey or connectivity
    (depending on whether stereochemistry data are being use or not). The second Series must be the group assignment
    of each compound specifying either 'training' or 'test' for each entry (other labels are disregarded) when
    `split_by` is 'custom' or cluster membership when `split_by` is 'custom-cluster'.
    :param scale: should the features be scaled using the custom scaling_method
    :param scale_method: scaling method to be applied to features (ignored if scale is False)
    :param yscramble: should the endpoint be shuffled to compare performance to the unshuffled endpoint
    :param random_state: seed to use for train/test splitting and KFold shuffling
    :param verbose: log details to stdout
    :param leave_level: whether the (single) progress bar stays on screen
        once done; ``0`` no, anything else (``-1`` default) yes
    :return: both:
    - a dataframe of the cross-validation results where each line is a fold of PCM modelling
    - a dictionary of the feature scaler (if used), label encoder (if mode is a classifier),
    the data splitter for cross-validation, fitted models on each cross-validation fold,
    the model fitted on the complete training set.
    """
    if model is None:
        model = xgboost.XGBRegressor(verbosity=0)
    if scale_method is None:
        scale_method = StandardScaler()
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    if split_by.lower() not in ['year', 'random', 'cluster', 'custom-cluster', 'custom']:
        raise ValueError("split not supported, must be one of {'Year', 'random', 'cluster', "
                         "'custom-cluster', 'custom'}")
    if split_by.lower() == 'cluster' and cluster_method is None:
        raise ValueError("cluster_method must be given if split_by is 'cluster'")
    if split_by.lower() in ('custom-cluster', 'custom') and custom_groups is None:
        raise ValueError("custom_groups must be given if split_by is 'custom-cluster' or 'custom'")
    if not isinstance(model, (RegressorMixin, ClassifierMixin)):
        raise ValueError('model type can only be a Scikit-Learn compliant regressor or classifier')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if isinstance(model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
        warnings.filterwarnings("ignore", category=UserWarning)
    model_type = 'regressor' if isinstance(model, RegressorMixin) else 'classifier'
    # Keep only required fields
    merge_on = 'connectivity' if 'connectivity' in data.columns else 'InChIKey'
    if model_type == 'regressor':
        features_to_ignore = [merge_on, 'target_id', endpoint, 'Year']
        data = data[data['relation'] == '='][features_to_ignore]
    else:
        features_to_ignore = [merge_on, 'target_id', 'Activity_class', 'Year']
        preserved = data[~data['Activity_class'].isna()]
        preserved = preserved.drop(
            columns=[col for col in preserved if col not in [merge_on, 'target_id', 'Activity_class', 'Year']])
        active = data[data['Activity_class'].isna() & (data[endpoint] > activity_threshold)]
        active = active[~active['relation'].str.contains('<')][features_to_ignore].copy()
        active.loc[:, 'Activity_class'] = 'A'
        inactive = data[data['Activity_class'].isna() & (data[endpoint] <= activity_threshold)]
        inactive = inactive[~inactive['relation'].str.contains('>')][features_to_ignore].copy()
        inactive.loc[:, 'Activity_class'] = 'N'
        data = pd.concat([preserved, active, inactive])
        # Change endpoint
        endpoint = 'Activity_class'
        del preserved, active, inactive
    # Get and merge molecular descriptors
    mol_descs = read_molecular_descriptors(mol_descriptors, 'connectivity' not in data.columns,
                                           version, mol_descriptor_chunksize, mol_descriptor_path)
    mol_descs = filter_molecular_descriptors(mol_descs, merge_on, data[merge_on].unique())
    data = data.merge(mol_descs, on=merge_on)
    merge_on_values = data[[merge_on]]
    data = data.drop(columns=[merge_on])
    # Get and merge protein descriptors
    prot_descs = read_protein_descriptors(prot_descriptors, version, prot_descriptor_chunksize,
                                          prot_sequences_path if isinstance(prot_descriptors, (Descriptor, Transform))
                                          else prot_descriptor_path,
                                          data['target_id'].unique())
    if isinstance(prot_descs, pl.LazyFrame):
        prot_descs = prot_descs.collect()
    if isinstance(prot_descs, pl.DataFrame):
        prot_descs = prot_descs.to_pandas()
    data = data.merge(prot_descs, on='target_id')
    data = data.drop(columns=['target_id'])
    del prot_descs
    # Build PCM model for targets reaching criteria
    # Insufficient data points
    if data.shape[0] < num_points:
        raise ValueError(f'too few datapoints to build PCM model: {data.shape[0]} while at least {num_points} expected')
    if model_type == 'regressor':
        min_activity = data[endpoint].min()
        max_activity = data[endpoint].max()
        delta = max_activity - min_activity
        # Not enough activity amplitude
        if delta < delta_activity:
            raise ValueError(f'amplitude of activity to narrow: {delta} while at least {delta_activity} expected')
    # Split, fit and evaluate - see _fit_and_evaluate's docstring for the
    # pipeline shared with qsar()'s per-target loop.
    try:
        performance, return_val, cv_models = _fit_and_evaluate(
            data, endpoint, model_type, model, merge_on, merge_on_values,
            split_by, split_year, test_set_size, cluster_method, custom_groups,
            features_to_ignore, ['Year'], scale, scale_method,
            yscramble, stratify, folds, random_state, verbose,
            strict_split_checks=False, cv_leave=_leave_bar(leave_level, 1),
        )
    except _InsufficientDataError as exc:
        raise ValueError(str(exc)) from None
    full_model = cv_models['Full model']
    # Set warnings back to default
    if isinstance(full_model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
        warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    return_val = {**return_val, **cv_models}
    return performance, return_val

