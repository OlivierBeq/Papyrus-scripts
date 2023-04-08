# -*- coding: utf-8 -*-

"""Modelling capacities of the Papyrus-scripts from the Papyrus dataset."""

from copy import deepcopy
import warnings
from itertools import chain, combinations
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import (pearsonr as pearsonR,
                         spearmanr as spearmanR,
                         kendalltau as kendallTau)

from tqdm.auto import tqdm

import xgboost
from sklearn.base import RegressorMixin, ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.model_selection import train_test_split, BaseCrossValidator, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (r2_score as R2,
                             mean_squared_error as MSE,
                             roc_auc_score as ROCAUC,
                             confusion_matrix,
                             multilabel_confusion_matrix,
                             matthews_corrcoef as MCC,
                             explained_variance_score as eVar,
                             max_error as maxE,
                             mean_absolute_error as MAE,
                             mean_squared_log_error as MSLE,
                             mean_poisson_deviance as MPD,
                             mean_gamma_deviance as MGD,
                             )

from prodec.Descriptor import Descriptor
from prodec.Transform import Transform

from .reader import read_molecular_descriptors, read_protein_descriptors
from .preprocess import yscrambling
from .neuralnet import (BaseNN,
                        SingleTaskNNClassifier,
                        SingleTaskNNRegressor,
                        MultiTaskNNRegressor,
                        MultiTaskNNClassifier
                        )

pd.set_option('mode.chained_assignment', None)


def filter_molecular_descriptors(data: Union[pd.DataFrame, Iterator],
                                 column_name: str,
                                 keep_values: Iterable,
                                 progress: bool = True,
                                 total: Optional[int] = None) -> pd.DataFrame:
    """Filter the data so that the desired column contains only the desired data.

    :param data: data to be filtered, either a dataframe or an iterator of chunks
    :param column_name: name of the column to apply the filter on
    :param keep_values: allowed values
    :param progress: show progress bar
    :param total: number of chunks in data if data is an Iterator
    :return: a pandas dataframe
    """
    if isinstance(data, pd.DataFrame):
        return data[data[column_name].isin(keep_values)]
    elif progress:
        return pd.concat([chunk[chunk[column_name].isin(keep_values)]
                          for chunk in tqdm(data, total=total, desc='Loading molecular descriptors')],
                         axis=0)
    else:
        return pd.concat([chunk[chunk[column_name].isin(keep_values)]
                          for chunk in data],
                         axis=0)


def model_metrics(model, y_true, x_test) -> dict:
    """Determine performance metrics of a model

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
        # Slope of predicted vs observed
        k = sum(xi * yi for xi, yi in zip(y_true, y_pred)) / sum(xi ** 2 for xi in y_true)
        # Slope of observed vs predicted
        k_prime = sum(xi * yi for xi, yi in zip(y_true, y_pred)) / sum(yi ** 2 for yi in y_pred)
        # Mean averages
        y_true_mean = y_true.mean()
        y_pred_mean = y_pred.mean()
        return {'number': y_true.size,
                'R2': R2(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'MSE': MSE(y_true, y_pred, squared=True) if len(y_pred) >= 2 else 0,
                'RMSE': MSE(y_true, y_pred, squared=False) if len(y_pred) >= 2 else 0,
                'MSLE': MSLE(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'RMSLE': np.sqrt(MSLE(y_true, y_pred)) if len(y_pred) >= 2 else 0,
                'MAE': MAE(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Explained Variance': eVar(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Max Error': maxE(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Mean Poisson Distrib': MPD(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Mean Gamma Distrib': MGD(y_true, y_pred) if len(y_pred) >= 2 else 0,
                'Pearson r': pearsonR(y_true, y_pred)[0] if len(y_pred) >= 2 else 0,
                'Spearman r': spearmanR(y_true, y_pred)[0] if len(y_pred) >= 2 else 0,
                'Kendall tau': kendallTau(y_true, y_pred)[0] if len(y_pred) >= 2 else 0,
                'R2_0 (pred. vs. obs.)': 1 - (sum((xi - k_prime * yi) ** 2
                                                  for xi, yi in zip(y_true, y_pred)) /
                                              sum((xi - y_true_mean) ** 2
                                                  for xi in y_true)) if len(y_pred) >= 2 else 0,
                'R\'2_0 (obs. vs. pred.)': 1 - (sum((yi - k * xi) ** 2
                                                    for xi, yi in zip(y_true, y_pred)) /
                                                sum((yi - y_pred_mean) ** 2
                                                    for yi in y_pred)) if len(y_pred) >= 2 else 0,
                'k slope (pred. vs obs.)': k,
                'k\' slope (obs. vs pred.)': k_prime,
                }
    # Classification
    elif isinstance(model, (ClassifierMixin, SingleTaskNNClassifier, MultiTaskNNClassifier)):
        # Binary classification
        if len(model.classes_) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=model.classes_).ravel()
            values = {}
            try:
                mcc = MCC(y_true, y_pred)
                values['MCC'] = mcc
            except RuntimeWarning:
                pass
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
            if hasattr(model, "predict_proba"):  # able to predict probability
                y_probas = model.predict_proba(x_test)
                if y_probas.shape[1] == 1:
                    y_proba = y_probas.ravel()
                    values['AUC 1'] = ROCAUC(y_true, y_proba)
                else:
                    for i in range(len(model.classes_)):
                        y_proba = y_probas[:, i].ravel()
                        try:
                            values['AUC %s' % model.classes_[i]] = ROCAUC(y_true, y_proba)
                        except ValueError:
                            warnings.warn('Only one class present in y_true. '
                                          'ROC AUC score is not defined in that case. '
                                          'Stratify your folds to avoid such warning.')
                            values['AUC %s' % model.classes_[i]] = np.nan
        # Multiclasses
        else:
            i = 0
            values = {}
            for contingency_matrix in multilabel_confusion_matrix(y_true, y_pred):
                tn, fp, fn, tp = contingency_matrix.ravel()
                try:
                    mcc = MCC(y_true, y_pred)
                    values['%s|MCC' % model.classes_[i]] = mcc
                except RuntimeWarning:
                    pass
                values['%s|number' % model.classes_[i]] = int(sum(y_true == model.classes_[i]))
                values['%s|ACC' % model.classes_[i]] = (tp + tn) / (tp + tn + fp + fn) \
                    if (tp + tn + fp + fn) != 0\
                    else 0
                values['%s|BACC' % model.classes_[i]] = (tp / (tp + fn) + tn / (tn + fp)) / 2
                values['%s|Sensitivity' % model.classes_[i]] = tp / (tp + fn) if tp + fn != 0 else 0
                values['%s|Specificity' % model.classes_[i]] = tn / (tn + fp) if tn + fp != 0 else 0
                values['%s|PPV' % model.classes_[i]] = tp / (tp + fp) if tp + fp != 0 else 0
                values['%s|NPV' % model.classes_[i]] = tn / (tn + fn) if tn + fn != 0 else 0
                values['%s|F1' % model.classes_[i]] = \
                    2 * values['%s|Sensitivity' % model.classes_[i]] * values['%s|PPV' % model.classes_[i]] / \
                    (values['%s|Sensitivity' % model.classes_[i]] + values['%s|PPV' % model.classes_[i]]) \
                     if (values['%s|Sensitivity' % model.classes_[i]] + values['%s|PPV' % model.classes_[i]]) != 0 \
                     else 0
                i += 1
            if hasattr(model, "predict_proba"):  # able to predict probability
                y_probas = model.predict_proba(x_test)
                try:
                    values['AUC 1 vs 1'] = ROCAUC(y_true, y_probas, average="macro", multi_class="ovo")
                    values['AUC 1 vs All'] = ROCAUC(y_true, y_probas, average="macro", multi_class="ovr")
                except ValueError:
                    warnings.warn('Only one class present in y_true. ROC AUC score is not defined in that case. '
                                  'Stratify your folds to avoid such warning.')
                    values['AUC 1 vs 1'] = np.nan
                    values['AUC 1 vs All'] = np.nan
        return values
    else:
        raise ValueError('model can only be classifier or regressor.')


def crossvalidate_model(data: pd.DataFrame,
                        model: Union[RegressorMixin, ClassifierMixin],
                        folds: BaseCrossValidator,
                        groups: List[int] = None,
                        verbose: bool = False
                        ) -> Tuple[pd.DataFrame, Dict[str, Union[RegressorMixin, ClassifierMixin]]]:
    """Create a machine learning model predicting values in the first column

   :param data: data containing the dependent vairable (in the first column) and other features
   :param model: estimator (either classifier or regressor) to use for model building
   :param folds: cross-validator
   :param groups: groups to split the labels according to
   :param verbose: whether to show fold progression
   :return: cross-validated performance and model trained on the entire dataset
    """
    X, y = data.iloc[:, 1:], data.iloc[:, 0].values.ravel()
    performance = []
    if verbose:
        pbar = tqdm(desc='Fitting model', total=folds.n_splits + 1)
    models = {}
    # Perform cross-validation
    for i, (train, test) in enumerate(folds.split(X, y, groups)):
        if verbose:
            pbar.set_description(f'Fitting model on fold {i + 1}', refresh=True)
        model.fit(X.iloc[train, :], y[train])
        models[f'Fold {i + 1}'] = deepcopy(model)
        performance.append(model_metrics(model, y[test], X.iloc[test, :]))
        if verbose:
            pbar.update()
    # Organize result in a dataframe
    performance = pd.DataFrame(performance)
    performance.index = [f'Fold {i + 1}' for i in range(folds.n_splits)]
    # Add average and sd of  performance
    performance.loc['Mean'] = [np.mean(performance[col]) if ':' not in col else '-' for col in performance]
    performance.loc['SD'] = [np.std(performance[col]) if ':' not in col else '-' for col in performance]
    # Fit model on the entire dataset
    if verbose:
        pbar.set_description('Fitting model on entire training set', refresh=True)
    model.fit(X, y)
    models['Full model'] = deepcopy(model)
    if verbose:
        pbar.update()
    return performance, models


def train_test_proportional_group_split(data: pd.DataFrame,
                                        groups: List[int],
                                        test_size: float = 0.30,
                                        verbose: bool = False
                                        ) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """Split the data into training and test sets according to the groups that respect most test_size

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
    best, proportion = min(zip(permutations, proportions), key=lambda x: (x[1] - test_size) ** 2)
    del counts, permutations, proportions
    if verbose:
        print(f'Best group permutation corresponds to {proportion:.2%} of the data')
    # Get test set assignment
    assignment = np.where(group in best for group in groups)
    opposite = np.logical_not(assignment)
    # Get training groups
    t_groups = [x for x in groups if x not in best]
    return data[opposite], data[assignment], t_groups, best


def qsar(data: pd.DataFrame,
         endpoint: str = 'pchembl_value_Mean',
         num_points: int = 30,
         delta_activity: float = 2,
         version: str = 'latest',
         descriptors: str = 'mold2',
         descriptor_path: Optional[str] = None,
         descriptor_chunksize: Optional[int] = 50000,
         activity_threshold: float = 6.5,
         model: Union[RegressorMixin, ClassifierMixin] = xgboost.XGBRegressor(verbosity=0),
         folds: int = 5,
         stratify: bool = False,
         split_by: str = 'Year',
         split_year: int = 2013,
         test_set_size: float = 0.30,
         cluster_method: ClusterMixin = None,
         custom_groups: pd.DataFrame = None,
         scale: bool = False,
         scale_method: TransformerMixin = StandardScaler(),
         yscramble: bool = False,
         random_state: int = 1234,
         verbose: bool = True
         ) -> Tuple[pd.DataFrame, Dict[str,
                                       Optional[Union[TransformerMixin, LabelEncoder,
                                                      BaseCrossValidator, Dict[str, ClassifierMixin]]]]]:
    """Create QSAR models for as many targets with selected data source(s),
    data quality, minimum number of datapoints and minimum activity amplitude.

    :param data: Papyrus activity data
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
    :param split_by: how should folds be determined {'random', 'Year', 'cluster', 'custom'}
    If 'random', exactly test_set_size is extracted for test set.
    If 'Year', the size of the test and training set are not looked at
    If 'cluster' or 'custom', the groups giving proportion closest to test_set_size will be used to defined the test set
    :param split_year: Year from which on the test set is extracted (ignored if split_by is not 'Year')
    :param test_set_size: proportion of the dataset to be used as test set
    :param cluster_method: clustering method to use to extract test set and cross-validation folds
    (ignored if split_by is not 'cluster')
    :param custom_groups: custom groups to use to extract test set and cross-validation fold
    (ignored if split_by is not 'custom').
    Groups must be a pandas DataFrame with only two Series. The first Series is either InChIKey or connectivity
    (depending on whether stereochemistry data are being use or not). The second Series must be the group assignment
    of each compound.
    :param scale: should the features be scaled using the custom scaling_method
    :param scale_method: scaling method to be applied to features (ignored if scale is False)
    :param yscramble: should the endpoint be shuffled to compare performance to the unshuffled endpoint
    :param random_state: seed to use for train/test splitting and KFold shuffling
    :param verbose: log details to stdout
    :return: both:
    - a dataframe of the cross-validation results where each line is a fold of QSAR modelling of an accession
    - a dictionary of the feature scaler (if used), label encoder (if mode is a classifier),
    the data splitter for cross-validation,  and for each accession in the data:
    the fitted models on each cross-validation fold and the model fitted on the complete training set.
    """
    if split_by.lower() not in ['year', 'random', 'cluster', 'custom']:
        raise ValueError("split not supported, must be one of {'Year', 'random', 'cluster', 'custom'}")
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
        active = active[~active['relation'].str.contains('<')][features_to_ignore]
        active.loc[:, 'Activity_class'] = 'A'
        inactive = data[data['Activity_class'].isna() & (data[endpoint] <= activity_threshold)]
        inactive = inactive[~inactive['relation'].str.contains('>')][features_to_ignore]
        inactive.loc[:, 'Activity_class'] = 'N'
        data = pd.concat([preserved, active, inactive])
        # Change endpoint
        endpoint = 'Activity_class'
        del preserved, active, inactive
    # Get  and merge molecular descriptors
    descs = read_molecular_descriptors(descriptors, 'connectivity' not in data.columns,
                                       version, descriptor_chunksize, descriptor_path)
    descs = filter_molecular_descriptors(descs, merge_on, data[merge_on].unique())
    data = data.merge(descs, on=merge_on)
    data = data.drop(columns=[merge_on])
    del descs
    # Table of results
    results, models = [], {}
    targets = list(data['target_id'].unique())
    n_targets = len(targets)
    if verbose:
        pbar = tqdm(total=n_targets, smoothing=0.1)
    # Build QSAR model for targets reaching criteria
    for i_target in range(n_targets - 1, -1, -1):
        tmp_data = data[data['target_id'] == targets[i_target]]
        if verbose:
            pbar.set_description(f'Building QSAR for target: {targets[i_target]} #datapoints {tmp_data.shape[0]}',
                                 refresh=True)
        # Insufficient data points
        if tmp_data.shape[0] < num_points:
            if model_type == 'regressor':
                results.append(pd.DataFrame([[targets[i_target],
                                              tmp_data.shape[0],
                                              f'Number of points {tmp_data.shape[0]} < {num_points}']],
                                            columns=['target', 'number', 'error']))
            else:
                data_classes = Counter(tmp_data[endpoint])
                results.append(
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
                results.append(pd.DataFrame([[targets[i_target],
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
                results.append(
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
                results.append(
                    pd.DataFrame([[targets[i_target],
                                   ':'.join(str(data_classes.get(x, 0)) for x in ['A', 'N']),
                                   f'Not enough data in minority class for all {folds} folds']],
                                 columns=['target', 'A:N', 'error']))
                if verbose:
                    pbar.update()
                models[targets[i_target]] = None
                continue
        # Set groups for fold enumerator and extract test set
        if split_by.lower() == 'year':
            groups = tmp_data['Year']
            test_set = tmp_data[tmp_data['Year'] >= split_year]
            if test_set.empty:
                if model_type == 'regressor':
                    results.append(pd.DataFrame([[targets[i_target],
                                                  tmp_data.shape[0],
                                                  f'No test data for temporal split at {split_year}']],
                                                columns=['target', 'number', 'error']))
                else:
                    data_classes = Counter(tmp_data[endpoint])
                    results.append(
                        pd.DataFrame([[targets[i_target],
                                       ':'.join(str(data_classes.get(x, 0)) for x in ['A', 'N']),
                                       f'No test data for temporal split at {split_year}']],
                                     columns=['target', 'A:N', 'error']))
                if verbose:
                    pbar.update()
                models[targets[i_target]] = None
                continue
            training_set = tmp_data[~tmp_data.index.isin(test_set.index)]
            if training_set.empty or training_set.shape[0] < folds:
                if model_type == 'regressor':
                    results.append(pd.DataFrame([[targets[i_target],
                                                  tmp_data.shape[0],
                                                  f'Not enough training data for temporal split at {split_year}']],
                                                columns=['target', 'number', 'error']))
                else:
                    data_classes = Counter(tmp_data[endpoint])
                    results.append(
                        pd.DataFrame([[targets[i_target],
                                       ':'.join(str(data_classes.get(x, 0)) for x in ['A', 'N']),
                                       f'Not enough training data for temporal split at {split_year}']],
                                     columns=['target', 'A:N', 'error']))
                if verbose:
                    pbar.update()
                models[targets[i_target]] = None
                continue
            if model_type == 'classifier':
                train_data_classes = Counter(training_set[endpoint])
                test_data_classes = Counter(test_set[endpoint])
                if len(train_data_classes) < 2:
                    results.append(pd.DataFrame([[targets[i_target],
                                                  ':'.join(str(train_data_classes.get(x, 0)) for x in ['A', 'N']),
                                                  'Only one activity class in traing set '
                                                  f'for temporal split at {split_year}']],
                                                columns=['target', 'A:N', 'error']))
                    if verbose:
                        pbar.update()
                    continue
                elif len(test_data_classes) < 2:
                    results.append(pd.DataFrame([[targets[i_target],
                                                  ':'.join(str(test_data_classes.get(x, 0)) for x in ['A', 'N']),
                                                  'Only one activity class in traing set '
                                                  f'for temporal split at {split_year}']],
                                                columns=['target', 'A:N', 'error']))
                    if verbose:
                        pbar.update()
                    models[targets[i_target]] = None
                    continue
            training_groups = training_set['Year']
        elif split_by.lower() == 'random':
            training_groups = None
            training_set, test_set = train_test_split(tmp_data, test_size=test_set_size, random_state=random_state)
        elif split_by.lower() == 'cluster':
            groups = cluster_method.fit_predict(tmp_data.drop(columns=features_to_ignore))
            training_set, test_set, training_groups, _ = train_test_proportional_group_split(tmp_data, groups,
                                                                                             test_set_size,
                                                                                             verbose=verbose)
        elif split_by.lower() == 'custom':
            # Merge from custom split DataFrame
            groups = tmp_data[[merge_on]].merge(custom_groups, on=merge_on).iloc[:, 1].tolist()
            training_set, test_set, training_groups, _ = train_test_proportional_group_split(tmp_data, groups,
                                                                                             test_set_size,
                                                                                             verbose=verbose)
        # Drop columns not used for training
        training_set = training_set.drop(columns=['Year', 'target_id'])
        test_set = test_set.drop(columns=['Year', 'target_id'])
        X_train, y_train = training_set.drop(columns=[endpoint]), training_set.loc[:, endpoint]
        X_test, y_test = test_set.drop(columns=[endpoint]), test_set.loc[:, endpoint]
        # Scale data
        if scale:
            X_train.loc[X_train.index, X_train.columns] = scale_method.fit_transform(X_train)
            X_test.loc[X_test.index, X_test.columns] = scale_method.transform(X_test)
        # Encode labels
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
        # Y-scrambling
        if yscramble:
            training_set = yscrambling(data=training_set, y_var=endpoint, random_state=random_state)
            test_set = yscrambling(data=test_set, y_var=endpoint, random_state=random_state)
        # Make sure enough data
        if model_type == 'classifier':
            train_data_classes = Counter(training_set['Activity_class'])
            train_enough_data = np.all(np.array(list(train_data_classes.values())) > folds)
            test_data_classes = Counter(test_set['Activity_class'])
            test_enough_data = np.all(np.array(list(test_data_classes.values())) > folds)
            if not train_enough_data:
                results.append(pd.DataFrame([[targets[i_target],
                                              ':'.join(str(train_data_classes.get(x, 0)) for x in ['A', 'N']),
                                              'Not enough data in minority class of '
                                              f'the training set for all {folds} folds']],
                                            columns=['target', 'A:N', 'error']))
                if verbose:
                    pbar.update()
                models[targets[i_target]] = None
                continue
            elif not test_enough_data:
                results.append(pd.DataFrame([[targets[i_target],
                                              ':'.join(str(test_data_classes.get(x, 0)) for x in ['A', 'N']),
                                              'Not enough data in minority class of '
                                              f'the training set for all {folds} folds']],
                                            columns=['target', 'A:N', 'error']))
                if verbose:
                    pbar.update()
                models[targets[i_target]] = None
                continue
        # Define folding scheme for cross validation
        if stratify and model_type == 'classifier':
            kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        else:
            kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        performance, cv_models = crossvalidate_model(training_set, model, kfold, training_groups)
        full_model = cv_models['Full model']
        X_test, y_test = test_set.iloc[:, 1:], test_set.iloc[:, 0].values.ravel()
        performance.loc['Test set'] = model_metrics(full_model, y_test, X_test)
        performance.loc[:, 'target'] = targets[i_target]
        results.append(performance.reset_index())
        models[targets[i_target]] = cv_models
        if verbose:
            pbar.update()
    if isinstance(model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
        warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    # Formatting return values
    return_val = {}
    if scale:
        return_val['scaler'] = deepcopy(scale_method)
    if model_type == 'classifier':
        return_val['label_encoder'] = deepcopy(lblenc)
        if stratify:
            return_val['data_splitter'] = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        return_val['data_splitter'] = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    return_val = {**return_val, **models}
    if len(results) is False:
        return pd.DataFrame(), return_val
    results = pd.concat(results, axis=0).set_index(['target', 'index'])
    results.index.names = ['target', None]
    return results, return_val


def pcm(data: pd.DataFrame,
        endpoint: str = 'pchembl_value_Mean',
        num_points: int = 30,
        delta_activity: float = 2,
        version: str = 'latest',
        mol_descriptors: str = 'mold2',
        mol_descriptor_path: Optional[str] = None,
        mol_descriptor_chunksize: Optional[int] = 50000,
        prot_sequences_path: Optional[str] = None,
        prot_descriptors: Union[str, Descriptor, Transform] = 'unirep',
        prot_descriptor_path: Optional[str] = None,
        prot_descriptor_chunksize: Optional[int] = 50000,
        activity_threshold: float = 6.5,
        model: Union[RegressorMixin, ClassifierMixin] = xgboost.XGBRegressor(verbosity=0),
        folds: int = 5,
        stratify: bool = False,
        split_by: str = 'Year',
        split_year: int = 2013,
        test_set_size: float = 0.30,
        cluster_method: ClusterMixin = None,
        custom_groups: pd.DataFrame = None,
        scale: bool = False,
        scale_method: TransformerMixin = StandardScaler(),
        yscramble: bool = False,
        random_state: int = 1234,
        verbose: bool = True
        ) -> Tuple[pd.DataFrame, Dict[str,
                                      Union[TransformerMixin, LabelEncoder,
                                            BaseCrossValidator, RegressorMixin, ClassifierMixin]]]:
    """Create PCM models for as many targets with selected data source(s),
    data quality, minimum number of datapoints and minimum activity amplitude.

    :param data: Papyrus activity data
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
    :param split_by: how should folds be determined {'random', 'Year', 'cluster', 'custom'}
    If 'random', exactly test_set_size is extracted for test set.
    If 'Year', the size of the test and training set are not looked at
    If 'cluster' or 'custom', the groups giving proportion closest to test_set_size will be used to defined the test set
    :param split_year: Year from which on the test set is extracted (ignored if split_by is not 'Year')
    :param test_set_size: proportion of the dataset to be used as test set
    :param cluster_method: clustering method to use to extract test set and cross-validation folds
    (ignored if split_by is not 'cluster')
    :param custom_groups: custom groups to use to extract test set and cross-validation fold
    (ignored if split_by is not 'custom').
    Groups must be a pandas DataFrame with only two Series.The first Series is either InChIKey or connectivity
    (depending on whether stereochemistry data are being use or not). The second Series must be the group assignment
    of each compound.
    :param scale: should the features be scaled using the custom scaling_method
    :param scale_method: scaling method to be applied to features (ignored if scale is False)
    :param yscramble: should the endpoint be shuffled to compare performance to the unshuffled endpoint
    :param random_state: seed to use for train/test splitting and KFold shuffling
    :param verbose: log details to stdout
    :return: both:
    - a dataframe of the cross-validation results where each line is a fold of PCM modelling
    - a dictionary of the feature scaler (if used), label encoder (if mode is a classifier),
    the data splitter for cross-validation, fitted models on each cross-validation fold,
    the model fitted on the complete training set.
    """
    if split_by.lower() not in ['year', 'random', 'cluster', 'custom']:
        raise ValueError("split not supported, must be one of {'Year', 'random', 'cluster', 'custom'}")
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
        active = active[~active['relation'].str.contains('<')][features_to_ignore]
        active.loc[:, 'Activity_class'] = 'A'
        inactive = data[data['Activity_class'].isna() & (data[endpoint] <= activity_threshold)]
        inactive = inactive[~inactive['relation'].str.contains('>')][features_to_ignore]
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
    data = data.drop(columns=[merge_on])
    # Get and merge protein descriptors
    prot_descs = read_protein_descriptors(prot_descriptors, version, prot_descriptor_chunksize,
                                          prot_sequences_path if isinstance(prot_descriptors, (Descriptor, Transform))
                                          else prot_descriptor_path,
                                          data['target_id'].unique())
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
    # Set groups for fold enumerator and extract test set
    if split_by.lower() == 'year':
        groups = data['Year']
        test_set = data[data['Year'] >= split_year]
        if test_set.empty:
            raise ValueError(f'no test data for temporal split at {split_year}')
        training_set = data[~data.index.isin(test_set.index)]
        if training_set.empty:
            raise ValueError(f'no training data for temporal split at {split_year}')
        training_groups = training_set['Year']
    elif split_by.lower() == 'random':
        training_groups = None
        training_set, test_set = train_test_split(data, test_size=test_set_size, random_state=random_state)
    elif split_by.lower() == 'cluster':
        groups = cluster_method.fit_predict(data.drop(columns=features_to_ignore))
        training_set, test_set, training_groups, _ = train_test_proportional_group_split(data, groups,
                                                                                         test_set_size,
                                                                                         verbose=verbose)
    elif split_by.lower() == 'custom':
        # Merge from custom split DataFrame
        groups = data[[merge_on]].merge(custom_groups, on=merge_on).iloc[:, 1].tolist()
        training_set, test_set, training_groups, _ = train_test_proportional_group_split(data, groups,
                                                                                         test_set_size,
                                                                                         verbose=verbose)
    # Drop columns not used for training
    training_set = training_set.drop(columns=['Year'])
    test_set = test_set.drop(columns=['Year'])
    # Scale data
    X_train, y_train = training_set.drop(columns=[endpoint]), training_set.loc[:, endpoint]
    X_test, y_test = test_set.drop(columns=[endpoint]), test_set.loc[:, endpoint]
    if scale:
        X_train.loc[X_train.index, X_train.columns] = scale_method.fit_transform(X_train)
        X_test.loc[X_test.index, X_test.columns] = scale_method.transform(X_test)
    # Encode labels
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
    # Y-scrambling
    if yscramble:
        training_set = yscrambling(data=training_set, y_var=endpoint, random_state=random_state)
        test_set = yscrambling(data=test_set, y_var=endpoint, random_state=random_state)
    # Make sure enough data
    if model_type == 'classifier':
        enough_data = np.all(np.array(list(Counter(training_set['Activity_class']).values())) > folds)
        if not enough_data:
            raise ValueError(f'Too few data points for some classes: expected at least {folds} in total')
    # Define folding scheme for cross validation
    if stratify and model_type == 'classifier':
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    performance, cv_models = crossvalidate_model(training_set, model, kfold, training_groups, verbose=True)
    full_model = cv_models['Full model']
    X_test, y_test = test_set.iloc[:, 1:], test_set.iloc[:, 0].values.ravel()
    performance.loc['Test set'] = model_metrics(full_model, y_test, X_test)
    # Set warnings back to default
    if isinstance(full_model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
        warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    # Formatting return values
    return_val = {}
    if scale:
        return_val['scaler'] = deepcopy(scale_method)
    if model_type == 'classifier':
        return_val['label_encoder'] = deepcopy(lblenc)
        if stratify:
            return_val['data_splitter'] = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        return_val['data_splitter'] = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    return_val = {**return_val, **cv_models}
    return performance, return_val


# def dnn(data: pd.DataFrame,
#         endpoint: str = 'pchembl_value_Mean',
#         pcm: bool = False,
#         num_points: int = 30,
#         delta_activity: float = 2,
#         mol_descriptors: str = 'mold2',
#         mol_descriptor_path: str = './',
#         mol_descriptor_chunksize: Optional[int] = 50000,
#         prot_sequences_path: str = './',
#         prot_descriptors: Union[str, Descriptor, Transform] = 'unirep',
#         prot_descriptor_path: str = './',
#         prot_descriptor_chunksize: Optional[int] = 50000,
#         activity_threshold: float = 6.5,
#         model: Optional[BaseNN] = None,
#         folds: int = 5,
#         stratify: bool = False,
#         split_by: str = 'Year',
#         split_year: int = 2013,
#         validation_set_size: float = 0.20,
#         test_set_size: float = 0.30,
#         cluster_method: ClusterMixin = None,
#         custom_groups: pd.DataFrame = None,
#         scale: bool = True,
#         random_state: int = 1234,
#         verbose: bool = True
#         ) -> Tuple[pd.DataFrame, Union[RegressorMixin, ClassifierMixin]]:
#     """Create PCM models for as many targets with selected data source(s),
#     data quality, minimum number of datapoints and minimum activity amplitude.
#
#     :param data: Papyrus activity data
#     :param endpoint: value to be predicted or to derive classes from
#     :param pcm: should the DNN model be PCM model, otherwise QSAR
#     :param num_points: minimum number of points for the activity of a target to be modelled
#     :param delta_activity: minimum difference between most and least active compounds for a target to be modelled
#     :param mol_descriptors: type of desriptors to be used for model training
#     :param mol_descriptor_path: path to Papyrus descriptors
#     :param mol_descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded
#     (None disables chunking)
#     :param prot_sequences_path: path to Papyrus sequences
#     :param prot_descriptors: type of desriptors to be used for model training
#     :param prot_descriptor_path: path to Papyrus descriptors
#     :param prot_descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded
#     (None disables chunking)
#     :param activity_threshold: threshold activity between acvtive and inactive compounds
#     (ignored if using a regressor)
#     :param model: DNN model to be fitted (default: None = SingleTaskNNClassifier
#     :param folds: number of cross-validation folds to be performed
#     :param stratify: whether to stratify folds for cross validation, ignored if model is RegressorMixin
#     :param split_by: how should folds be determined {'random', 'Year', 'cluster', 'custom'}
#     If 'random', exactly test_set_size is extracted for test set.
#     If 'Year', the size of the test and training set are not looked at
#     If 'cluster' or 'custom', the groups giving proportion closest to test_set_size
#     will be used to defined the test set
#     :param split_year: Year from which on the test set is extracted (ignored if split_by is not 'Year')
#     :param test_set_size: proportion of the dataset to be used as test set
#     :param cluster_method: clustering method to use to extract test set and cross-validation folds
#     (ignored if split_by is not 'cluster')
#     :param custom_groups: custom groups to use to extract test set and cross-validation fold
#     (ignored if split_by is not 'custom').
#     Groups must be a pandas DataFrame with only two Series. The first Series is either InChIKey or connectivity
#     (depending on whether stereochemistry data are being use or not). The second Series must be the group assignment
#     of each compound.
#     :param scale: should to data be scaled to zero mean and unit variance
#     :param random_state: seed to use for train/test splitting and KFold shuffling
#     :param verbose: log details to stdout
#     :return: both:
#     - a dataframe of the cross-validation results where each line is a fold of PCM modelling
#     - the model fitted on all folds for further use
#     """
#     if split_by.lower() not in ['year', 'random', 'cluster', 'custom']:
#         raise ValueError("split not supported, must be one of {'Year', 'random', 'cluster', 'custom'}")
#     if not isinstance(model, (RegressorMixin, ClassifierMixin, SingleTaskNNClassifier,
#                               SingleTaskNNRegressor, MultiTaskNNClassifier, MultiTaskNNRegressor)):
#         raise ValueError('model type can only be a Scikit-Learn compliant regressor or classifier')
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     if model is None:
#         model = SingleTaskNNClassifier('./')
#     model_type = 'regressor' if isinstance(model, (SingleTaskNNRegressor, MultiTaskNNRegressor)) else 'classifier'
#     # Keep only required fields
#     merge_on = 'connectivity' if 'connectivity' in data.columns else 'InChIKey'
#     if model_type == 'regressor':
#         features_to_ignore = [merge_on, 'target_id', endpoint, 'Year']
#         data = data[data['relation'] == '='][features_to_ignore]
#         old_endpoint = []
#     else:
#         features_to_ignore = [merge_on, 'target_id', 'Activity_class', 'Year']
#         preserved = data[~data['Activity_class'].isna()]
#         preserved = preserved.drop(
#             columns=[col for col in preserved if col not in [merge_on, 'target_id', 'Activity_class', 'Year']])
#         active = data[data['Activity_class'].isna() & (data[endpoint] > activity_threshold)]
#         active = active[~active['relation'].str.contains('<')][features_to_ignore]
#         active.loc[:, 'Activity_class'] = 'A'
#         # active.drop(columns=[endpoint], inplace=True)
#         inactive = data[data['Activity_class'].isna() & (data[endpoint] <= activity_threshold)]
#         inactive = inactive[~inactive['relation'].str.contains('>')][features_to_ignore]
#         inactive.loc[:, 'Activity_class'] = 'N'
#         # inactive.drop(columns=[endpoint], inplace=True)
#         data = pd.concat([preserved, active, inactive])
#         # Change endpoint
#         endpoint = 'Activity_class'
#         del preserved, active, inactive
#     # Get and merge molecular descriptors
#     mol_descs = get_molecular_descriptors('connectivity' not in data.columns, mol_descriptors, mol_descriptor_path,
#                                           mol_descriptor_chunksize)
#     mol_descs = filter_molecular_descriptors(mol_descs, merge_on, data[merge_on].unique())
#     data = data.merge(mol_descs, on=merge_on)
#     data = data.drop(columns=[merge_on])
#     if pcm:
#         # Get and merge protein descriptors
#         prot_descs = get_protein_descriptors(prot_descriptors,
#                                              prot_sequences_path if isinstance(prot_descriptors,
#                                                                                (Descriptor, Transform))
#                                              else prot_descriptor_path,
#                                              prot_descriptor_chunksize,
#                                              data['target_id'].unique())
#         data = data.merge(prot_descs, on='target_id')
#         del prot_descs
#     # Transform for multi-task model
#     if isinstance(model, (MultiTaskNNRegressor, MultiTaskNNClassifier)):
#         targets = data['target_id'].unique()
#         data.loc[:, targets] = np.zeros((data.shape, len(targets)))
#         for target in targets:
#             mask = np.where(data.target_id == target)
#             data.loc[mask, target] = data.loc[mask, endpoint]
#         data = data.drop(columns=[endpoint])
#         endpoint = targets
#     data = data.drop(columns=['target_id'])
#     # Build model for targets reaching criteria
#     # Insufficient data points
#     if data.shape[0] < num_points:
#         raise ValueError('too few datapoints to build PCM model: '
#                          f'{data.shape[0]} while at least {num_points} expected')
#     if model_type == 'regressor':
#         min_activity = data[endpoint].min()
#         max_activity = data[endpoint].max()
#         delta = max_activity - min_activity
#         # Not enough activity amplitude
#         if delta < delta_activity:
#             raise ValueError(f'amplitude of activity to narrow: {delta} while at least {delta_activity} expected')
#     # Set groups for fold enumerator and extract test set
#     if split_by.lower() == 'year':
#         groups = data['Year']
#         test_set = data[data['Year'] >= split_year]
#         if test_set.empty:
#             raise ValueError(f'no test data for temporal split at {split_year}')
#         training_set = data[~data.index.isin(test_set.index)]
#         training_groups = training_set['Year']
#     elif split_by.lower() == 'random':
#         training_groups = None
#         training_set, test_set = train_test_split(data,
#                                                   test_size=test_set_size,
#                                                   random_state=random_state)
#     elif split_by.lower() == 'cluster':
#         groups = cluster_method.fit_predict(data.drop(columns=features_to_ignore))
#         training_set, test_set, training_groups, _ = train_test_proportional_group_split(data, groups,
#                                                                                          test_set_size,
#                                                                                          verbose=verbose)
#     elif split_by.lower() == 'custom':
#         # Merge from custom split DataFrame
#         groups = data[[merge_on]].merge(custom_groups, on=merge_on).iloc[:, 1].tolist()
#         training_set, test_set, training_groups, _ = train_test_proportional_group_split(data, groups,
#                                                                                          test_set_size,
#                                                                                          verbose=verbose)
#     training_set, validation_set = train_test_split(training_set,
#                                                     test_size=validation_set_size,
#                                                     random_state=random_state)
#     # Drop columns not used for training
#     training_set = training_set.drop(columns=['Year'])
#     validation_set = validation_set.drop(columns=['Year'])
#     test_set = test_set.drop(columns=['Year'])
#     # Scale data and reorganize
#     X_train, y_train = training_set.drop(columns=[endpoint]), training_set[[endpoint]]
#     X_validation, y_validation = validation_set.drop(columns=[endpoint]), validation_set[[endpoint]]
#     X_test, y_test = test_set.drop(columns=[endpoint]), test_set[[endpoint]]
#     if scale:
#         scaler = StandardScaler()
#         X_train.loc[X_train.index, X_train.columns] = scaler.fit_transform(X_train)
#         X_validation.loc[X_validation.index, X_validation.columns] = scaler.transform(X_validation)
#         X_test.loc[X_test.index, X_test.columns] = scaler.transform(X_test)
#     # Make sure enough data
#     if model_type == 'classifier':
#         enough_data = np.all(np.array(list(Counter(training_set['Activity_class']).values())) > folds)
#         if not enough_data:
#             print(Counter(training_set['Activity_class']))
#             raise ValueError(f'Too few data points for some classes: expected at least {folds} in total')
#     # Encode labels if not integers
#     lblenc = LabelEncoder()
#     y_train = lblenc.fit_transform(y_train.values.ravel())
#     y_test = lblenc.transform(y_test.values.ravel())
#     y_validation = lblenc.transform(y_validation.values.ravel())
#     y_test = y_test.astype(np.int32)
#     y_validation = y_validation.astype(np.int32)
#     # Combine sets
#     training_set = pd.concat([pd.Series(y_train), X_train], axis=1)
#     test_set = pd.concat([pd.Series(y_test), X_test], axis=1)
#     del y_train, X_test, y_test
#     # Define folding scheme for cross validation
#     if stratify and model_type == 'classifier':
#         kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
#     else:
#         kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
#     # Set validation set
#     model.set_validation(X_validation, y_validation)
#     # Set architecture
#     if isinstance(model, SingleTaskNNRegressor):
#         model.set_architecture(X_train.shape[1])
#     elif isinstance(model, SingleTaskNNClassifier):
#         model.set_architecture(X_train.shape[1], 1)
#     else:
#         model.set_architecture(X_train.shape[1], len(endpoint))
#     del X_train
#     performance, model = crossvalidate_model(training_set, model, kfold, verbose=True)
#     X_test, y_test = test_set.iloc[:, 1:], test_set.iloc[:, 0].values.ravel()
#     performance.loc['Test set'] = model_metrics(model, y_test, X_test)
#     warnings.filterwarnings("default", category=RuntimeWarning)
#     return performance, model
