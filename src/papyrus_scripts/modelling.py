# -*- coding: utf-8 -*-

import glob
import json
import os
import warnings
from functools import partial
from itertools import chain, combinations
from collections import Counter
from typing import Iterable, Iterator, List, Optional, Tuple, Union

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

from .utils.IO import TypeDecoder
from .reader import read_protein_set
from .neuralnet import (BaseNN,
                        SingleTaskNNClassifier,
                        SingleTaskNNRegressor,
                        MultiTaskNNRegressor,
                        MultiTaskNNClassifier
                        )


def get_molecular_descriptors(is3d: bool = False, desc_type: str = 'mold2', source_path: str = './',
                              chunksize: Optional[int] = None):
    """Get molecular descriptors

   :param desc_type: type of descriptor {'mold2', 'mordred', 'cddd', 'fingerprint', 'all'}
   :param is3d: whether to load descriptors of the dataset containing stereochemistry
   :param source_path: folder containing the molecular descriptor datasets
   :return: the dataframe of molecular descriptors
    """
    if desc_type not in ['mold2', 'mordred', 'cddd', 'fingerprint', 'all']:
        raise ValueError("descriptor type not supported")
    mold2_mask = os.path.join(source_path, f'*.*_combined_{3 if is3d else 2}D_moldescs_mold2.tsv*')
    mordd_mask = os.path.join(source_path, f'*.*_combined_{3 if is3d else 2}D_moldescs_mordred{3 if is3d else 2}D.tsv*')
    cddds_mask = os.path.join(source_path, f'*.*_combined_{3 if is3d else 2}D_moldescs_CDDDs.tsv*')
    molfp_mask = os.path.join(source_path,
                              f'*.*_combined_{3 if is3d else 2}D_moldescs_{"E3FP" if is3d else "ECFP6"}.tsv*')
    mold2_files = glob.glob(mold2_mask)
    mordd_files = glob.glob(mordd_mask)
    cddds_files = glob.glob(cddds_mask)
    molfp_files = glob.glob(molfp_mask)
    if desc_type in ['mold2', 'all'] and len(mold2_files) == 0:
        raise ValueError('Could not find Mold2 descriptor file')
    elif desc_type in ['mordred', 'all'] and len(mordd_files) == 0:
        raise ValueError('Could not find mordred descriptor file')
    elif desc_type in ['cddd', 'all'] and len(cddds_files) == 0:
        raise ValueError('Could not find CDDD file')
    elif desc_type in ['fingerprint', 'all'] and len(molfp_files) == 0:
        raise ValueError(f'Could not find {"E3FP" if is3d else "ECFP6"} file')
    # Load data types
    dtype_file = os.path.join(os.path.dirname(__file__), 'utils', 'data_types.json')
    with open(dtype_file, 'r') as jsonfile:
        dtypes = json.load(jsonfile, cls=TypeDecoder)
    if desc_type == 'mold2':
        return pd.read_csv(mold2_files[0], sep='\t', dtype=dtypes['mold2'], low_memory=True, chunksize=chunksize)
    elif desc_type == 'mordred':
        return pd.read_csv(mordd_files[0], sep='\t', dtype=dtypes[f'mordred_{3 if is3d else 2}D'], low_memory=True,
                           chunksize=chunksize)
    elif desc_type == 'cddd':
        return pd.read_csv(cddds_files[0], sep='\t', dtype=dtypes['CDDD'], low_memory=True, chunksize=chunksize)
    elif desc_type == 'fingerprint':
        return pd.read_csv(molfp_files[0], sep='\t', dtype=dtypes[f'{"E3FP" if is3d else "ECFP6"}'], low_memory=True,
                           chunksize=chunksize)
    elif desc_type == 'all':
        mold2 = pd.read_csv(mold2_files[0], sep='\t', dtype=dtypes['mold2'], low_memory=True, chunksize=chunksize)
        mordd = pd.read_csv(mordd_files[0], sep='\t', dtype=dtypes[f'mordred_{3 if is3d else 2}D'], low_memory=True,
                            chunksize=chunksize)
        cddds = pd.read_csv(cddds_files[0], sep='\t', dtype=dtypes['CDDD'], low_memory=True, chunksize=chunksize)
        molfp = pd.read_csv(molfp_files[0], sep='\t', dtype=dtypes[f'{"E3FP" if is3d else "ECFP6"}'], low_memory=True,
                            chunksize=chunksize)
        if chunksize is None:
            mold2.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            mordd.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            molfp.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            cddds.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            data = pd.concat([mold2, mordd, cddds, molfp], axis=1)
            del mold2, mordd, cddds, molfp
            data.reset_index(inplace=True)
            return data
        return _join_molecular_descriptors(mold2, mordd, molfp, cddds)


def get_protein_descriptors(desc_type: Union[str, Descriptor, Transform] = 'unirep', source_path: str = './',
                            chunksize: int = 50000, ids: Optional[List[str]] = None, verbose: bool = True):
    """Get protein descriptors

   :param desc_type: type of descriptor {'unirep'} or a prodec Descriptor or Transform
   :param source_path: If desc_type is 'unirep', folder containing the protein descriptor datasets.
                       If desc_type is 'custom', the file path to a dataframe containing target_id
                       as its first column and custom descriptors in the following ones.
                       If desc_type is a ProDEC Descriptor or Transform, folder containing Papyrus protein data.
   :param chunksize: size of chunks to be iteratively loaded, ignored if desc_type is not 'unirep'
   :param ids: identifiers of the sequences which descriptors should be loaded
   :param verbose: whether to show progress
   :return: the dataframe of protein descriptors
    """
    if desc_type not in ['unirep', 'custom'] and not isinstance(desc_type, (Descriptor, Transform)):
        raise ValueError("descriptor type not supported")
    if verbose:
        pbar = partial(tqdm, desc='Loading protein descriptors')
    else:
        pbar = partial(iter)
    if desc_type == 'unirep':
        unirep_mask = os.path.join(source_path, f'*.*_combined_prot_embeddings_unirep.tsv*')
        unirep_files = glob.glob(unirep_mask)
        if len(unirep_files) == 0:
            raise ValueError('Could not find unirep descriptor file')
        # Load data types
        dtype_file = os.path.join(os.path.dirname(__file__), 'utils', 'data_types.json')
        with open(dtype_file, 'r') as jsonfile:
            dtypes = json.load(jsonfile, cls=TypeDecoder)
        if desc_type == 'unirep':
            return pd.concat([chunk[chunk['target_id'].isin(ids)]
                              if 'target_id' in chunk.columns
                              else chunk[chunk['TARGET_NAME'].isin(ids)]
                              for chunk in pbar(pd.read_csv(unirep_files[0], sep='\t', dtype=dtypes['unirep'],
                                                       low_memory=True, chunksize=chunksize))
                              ]).rename(columns={'TARGET_NAME': 'target_id'})
    elif desc_type == 'custom':
        if not os.path.isfile(source_path):
            raise ValueError('source_path must be a file if using a custom descriptor type')
        return pd.concat([chunk[chunk['target_id'].isin(ids)]
                          if 'target_id' in chunk.columns
                          else chunk[chunk['TARGET_NAME'].isin(ids)]
                          for chunk in pbar(pd.read_csv(source_path, sep='\t', low_memory=True, chunksize=chunksize))
                          ]).rename(columns={'TARGET_NAME': 'target_id'})
    else:
        # Calculate protein descriptors
        protein_data = read_protein_set(source_path)
        protein_data = protein_data[protein_data['TARGET_NAME'].isin(ids)]
        descriptors = desc_type.pandas_get(protein_data['Sequence'], protein_data['TARGET_NAME'], quiet=True)
        descriptors.rename(columns={'ID': 'target_id'}, inplace=True)
        return descriptors


def _join_molecular_descriptors(*descriptors: Iterator, on: str = 'connectivity') -> Iterator:
    """Concatenate multiple types of molecular descriptors on the same identifier.

    :param descriptors: the different iterators of descriptors to be joined
    :param on: identifier to join the descriptors on
    """
    try:
        while True:
            values = [next(descriptor).set_index(on) for descriptor in descriptors]
            data = pd.concat(values, axis=1)
            data.reset_index(inplace=True)
            yield data
    except StopIteration:
        raise StopIteration


def filter_molecular_descriptors(data: Union[pd.DataFrame, Iterator],
                                 column_name: str,
                                 keep_values: Iterable,
                                 progress: bool = True,
                                 total: Optional[int] = None) -> pd.DataFrame:
    """Filter the data so that the desired column contains only the desired data.

    :param data: data to be filtered, either a dataframe or an iterator of chunks
    :param column_name: name of the column to apply the filter on
    :param keep_values: allowed values
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

   :param model: model to check the performance of
   :param y_true: true labels
   :param x_test: testing set of features
   :return: a dictionary of metrics
    """
    y_pred = model.predict(x_test)
    # Regression metrics
    if isinstance(model, (RegressorMixin, SingleTaskNNRegressor, MultiTaskNNRegressor)):
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
                'Kendall tau': kendallTau(y_true, y_pred)[0] if len(y_pred) >= 2 else 0
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
            values[':'.join(str(x) for x in model.classes_)] = ':'.join([str(int(sum(y_true == class_))) for class_ in model.classes_])
            values['ACC'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
            values['BACC'] = (tp / (tp + fn) + tn / (tn + fp)) / 2
            values['Sensitivity'] = tp / (tp + fn) if tp + fn != 0 else 0
            values['Specificity'] = tn / (tn + fp) if tn + fp != 0 else 0
            values['PPV'] = tp / (tp + fp) if tp + fp != 0 else 0
            values['NPV'] = tn / (tn + fn) if tn + fn != 0 else 0
            values['F1'] = 2 * values['Sensitivity'] * values['PPV'] / (values['Sensitivity'] + values['PPV']) if (values['Sensitivity'] + values['PPV']) != 0 else 0
            if hasattr(model, "predict_proba"):  # able to predict probability
                y_probas = model.predict_proba(x_test)
                if y_probas.shape[1] == 1:
                    y_proba = y_probas.ravel()
                    values['AUC 1'] = ROCAUC(y_true, y_probas)
                else:
                    for i in range(len(model.classes_)):
                        y_proba = y_probas[:, i].ravel()
                        try:
                            values['AUC %s' % model.classes_[i]] = ROCAUC(y_true, y_proba)
                        except ValueError:
                            warnings.warn('Only one class present in y_true. ROC AUC score is not defined in that case. '
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
                values['%s|ACC' % model.classes_[i]] = (tp + tn) / (tp + tn + fp + fn) if (
                                                                                                  tp + tn + fp + fn) != 0 else 0
                values['%s|BACC' % model.classes_[i]] = (tp / (tp + fn) + tn / (tn + fp)) / 2
                values['%s|Sensitivity' % model.classes_[i]] = tp / (tp + fn) if tp + fn != 0 else 0
                values['%s|Specificity' % model.classes_[i]] = tn / (tn + fp) if tn + fp != 0 else 0
                values['%s|PPV' % model.classes_[i]] = tp / (tp + fp) if tp + fp != 0 else 0
                values['%s|NPV' % model.classes_[i]] = tn / (tn + fn) if tn + fn != 0 else 0
                values['%s|F1' % model.classes_[i]] = 2 * values['%s|Sensitivity' % model.classes_[i]] * values[
                    '%s|PPV' % model.classes_[i]] / (values['%s|Sensitivity' % model.classes_[i]] + values[
                    '%s|PPV' % model.classes_[i]]) if (values['%s|Sensitivity' % model.classes_[i]] + values[
                    '%s|PPV' % model.classes_[i]]) != 0 else 0
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
                        model: Union[RegressorMixin, ClassifierMixin, BaseNN],
                        folds: BaseCrossValidator,
                        groups: List[int] = None,
                        verbose: bool = False
                        ) -> Tuple[pd.DataFrame, Union[RegressorMixin, ClassifierMixin]]:
    """Create a machine learning model predicting values in the first column

   :param data: data containing the dependent vairable (in the first column) and other features
   :param model: estimator (may be classifier or regressor) to use for model building
   :param folds: cross-validator
   :param groups: groups to split the labels according to
   :param verbose: whether to show fold progression
   :return: cross-validated performance and model trained on the entire dataset
    """
    X, y = data.iloc[:, 1:], data.iloc[:, 0].values.ravel()
    performance = []
    if verbose:
        pbar = tqdm(desc='Fitting model', total=folds.n_splits + 1)
    # Perform cross-validation
    for i, (train, test) in enumerate(folds.split(X, y, groups)):
        if verbose:
            pbar.set_description(f'Fitting model on fold {i + 1}', refresh=True)
        model.fit(X.iloc[train, :], y[train])
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
    if verbose:
        pbar.update()
    return performance, model


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
         descriptors: str = 'mold2',
         descriptor_path: str = './',
         descriptor_chunksize: Optional[int] = 50000,
         activity_threshold: float = 6.5,
         model: Union[RegressorMixin, ClassifierMixin] = xgboost.XGBRegressor(verbosity=0),
         folds: int = 5,
         stratify: bool = False,
         split_by: str = 'Year',
         split_year: int = 2013,
         test_set_size: float = 0.30,
         validation_set_size: float = 0.30,
         cluster_method: ClusterMixin = None,
         custom_groups: pd.DataFrame = None,
         scale: bool = False,
         scale_method: TransformerMixin = StandardScaler(),
         random_state: int = 1234,
         verbose: bool = True
         ) -> Tuple[pd.DataFrame, List[Union[RegressorMixin, ClassifierMixin]]]:
    """Create QSAR models for as many targets with selected data source(s),
    data quality, minimum number of datapoints and minimum activity amplitude.

    :param data: Papyrus activity data
    :param endpoint: value to be predicted or to derive classes from
    :param num_points: minimum number of points for the activity of a target to be modelled
    :param delta_activity: minimum difference between most and least active compounds for a target to be modelled
    :param descriptors: type of desriptors to be used for model training
    :param descriptor_path: path to Papyrus descriptors
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
    :param validation_set_size: proportion of the dataset to be used as validation set (ignored if model is not derived from BaseNN)
    :param cluster_method: clustering method to use to extract test set and cross-validation folds (ignored if split_by is not 'cluster')
    :param custom_groups: custom groups to use to extract test set and cross-validation fold (ignored if split_by is not 'custom').
                           Groups must be a pandas DataFrame with only two Series. The first Series is either InChIKey or connectivity
                           (depending on whether stereochemistry data are being use or not). The second Series must be the group assignment
                           of each compound.
    :param random_state: seed to use for train/test splitting and KFold shuffling
    :param verbose: log details to stdout
    :return: both:
                    - a dataframe of the cross-validation results where each line is a fold of QSAR modelling of an accession
                    - a list of the models fitted on all folds for further use
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
        # active.drop(columns=[endpoint], inplace=True)
        inactive = data[data['Activity_class'].isna() & (data[endpoint] <= activity_threshold)]
        inactive = inactive[~inactive['relation'].str.contains('>')][features_to_ignore]
        inactive.loc[:, 'Activity_class'] = 'N'
        # inactive.drop(columns=[endpoint], inplace=True)
        data = pd.concat([preserved, active, inactive])
        # Change endpoint
        endpoint = 'Activity_class'
        del preserved, active, inactive
    # Get  and merge molecular descriptors
    descs = get_molecular_descriptors('connectivity' not in data.columns, descriptors, descriptor_path,
                                      descriptor_chunksize)
    descs = filter_molecular_descriptors(descs, merge_on, data[merge_on].unique())
    data = data.merge(descs, on=merge_on)
    data = data.drop(columns=[merge_on])
    del descs
    # Table of results
    results, models = [], []
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
            # del targets[i_target]
            if verbose:
                pbar.update()
            models.append(None)
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
                # del targets[i_target]
                if verbose:
                    pbar.update()
                models.append(None)
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
                # del targets[i_target]
                if verbose:
                    pbar.update()
                models.append(None)
                continue
            # Not enough data in minority class for all folds
            elif not all(x >= folds for x in data_classes.values()):
                results.append(
                    pd.DataFrame([[targets[i_target],
                                   ':'.join(str(data_classes.get(x, 0)) for x in ['A', 'N']),
                                   f'Not enough data in minority class for all {folds} folds']],
                                 columns=['target', 'A:N', 'error']))
                # del targets[i_target]
                if verbose:
                    pbar.update()
                models.append(None)
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
                # warnings.warn(f'no test data for temporal split at {split_year} for target {targets[i_target]}')
                # del targets[i_target]
                if verbose:
                    pbar.update()
                models.append(None)
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
                # warnings.warn(f'no training data for temporal split at {split_year} for target {targets[i_target]}')
                # del targets[i_target]
                if verbose:
                    pbar.update()
                models.append(None)
                continue
            if model_type == 'classifier':
                train_data_classes = Counter(training_set[endpoint])
                test_data_classes = Counter(test_set[endpoint])
                if len(train_data_classes) < 2:
                    results.append(pd.DataFrame([[targets[i_target],
                                                  ':'.join(str(train_data_classes.get(x, 0)) for x in ['A', 'N']),
                                                  f'Only one activity class in traing set for temporal split at {split_year}']],
                                                columns=['target', 'A:N', 'error']))
                    # warnings.warn(f'unique class for temporal split at {split_year} for target {targets[i_target]}')
                    # del targets[i_target]
                    if verbose:
                        pbar.update()
                    continue
                elif len(test_data_classes) < 2:
                    results.append(pd.DataFrame([[targets[i_target],
                                                  ':'.join(str(test_data_classes.get(x, 0)) for x in ['A', 'N']),
                                                  f'Only one activity class in traing set for temporal split at {split_year}']],
                                                columns=['target', 'A:N', 'error']))
                    # warnings.warn(f'unique class for temporal split at {split_year} for target {targets[i_target]}')
                    # del targets[i_target]
                    if verbose:
                        pbar.update()
                    models.append(None)
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
        X_train, y_train = training_set.drop(columns=[endpoint]), training_set[[endpoint]]
        X_test, y_test = test_set.drop(columns=[endpoint]), test_set[[endpoint]]
        # Scale data and reorganize
        if scale:
            X_train.loc[X_train.index, X_train.columns] = scale_method.fit_transform(X_train)
            X_test.loc[X_test.index, X_test.columns] = scale_method.transform(X_test)
        training_set = pd.concat([y_train, X_train], axis=1)
        test_set = pd.concat([y_test, X_test], axis=1)
        del X_train, y_train, X_test, y_test
        # Make sure enough data
        if model_type == 'classifier':
            train_data_classes = Counter(training_set['Activity_class'])
            train_enough_data = np.all(np.array(list(train_data_classes.values())) > folds)
            test_data_classes = Counter(test_set['Activity_class'])
            test_enough_data = np.all(np.array(list(test_data_classes.values())) > folds)
            if not train_enough_data:
                results.append(pd.DataFrame([[targets[i_target],
                                              ':'.join(str(train_data_classes.get(x, 0)) for x in ['A', 'N']),
                                              f'Not enough data in minority class of the training set for all {folds} folds']],
                                            columns=['target', 'A:N', 'error']))
                # del targets[i_target]
                if verbose:
                    pbar.update()
                models.append(None)
                continue
            elif not test_enough_data:
                results.append(pd.DataFrame([[targets[i_target],
                                              ':'.join(str(test_data_classes.get(x, 0)) for x in ['A', 'N']),
                                              f'Not enough data in minority class of the training set for all {folds} folds']],
                                            columns=['target', 'A:N', 'error']))
        # Define folding scheme for cross validation
        if stratify and model_type == 'classifier':
            kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        else:
            kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        performance, model = crossvalidate_model(training_set, model, kfold, training_groups)
        X_test, y_test = test_set.iloc[:, 1:], test_set.iloc[:, 0].values.ravel()
        performance.loc['Test set'] = model_metrics(model, y_test, X_test)
        performance.loc[:, 'target'] = targets[i_target]
        results.append(performance.reset_index())
        models.append(model)
        if verbose:
            pbar.update()
    if isinstance(model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
        warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    if len(results) is False:
        return pd.DataFrame(), models
    results = pd.concat(results, axis=0).set_index(['target', 'index'])
    results.index.names = ['target', None]
    return results, models


def pcm(data: pd.DataFrame,
        endpoint: str = 'pchembl_value_Mean',
        num_points: int = 30,
        delta_activity: float = 2,
        mol_descriptors: str = 'mold2',
        mol_descriptor_path: str = './',
        mol_descriptor_chunksize: Optional[int] = 50000,
        prot_sequences_path: str = './',
        prot_descriptors: Union[str, Descriptor, Transform] = 'unirep',
        prot_descriptor_path: str = './',
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
        random_state: int = 1234,
        verbose: bool = True
        ) -> Tuple[pd.DataFrame, Union[RegressorMixin, ClassifierMixin]]:
    """Create PCM models for as many targets with selected data source(s),
    data quality, minimum number of datapoints and minimum activity amplitude.

    :param data: Papyrus activity data
    :param endpoint: value to be predicted or to derive classes from
    :param num_points: minimum number of points for the activity of a target to be modelled
    :param delta_activity: minimum difference between most and least active compounds for a target to be modelled
    :param mol_descriptors: type of desriptors to be used for model training
    :param mol_descriptor_path: path to Papyrus descriptors
    :param mol_descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded (None disables chunking)
    :param prot_sequences_path: path to Papyrus sequences
    :param prot_descriptors: type of desriptors to be used for model training
    :param prot_descriptor_path: path to Papyrus descriptors
    :param prot_descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded (None disables chunking)
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
    :param cluster_method: clustering method to use to extract test set and cross-validation folds (ignored if split_by is not 'cluster')
    :param custom_groups: custom groups to use to extract test set and cross-validation fold (ignored if split_by is not 'custom').
                           Groups must be a pandas DataFrame with only two Series. The first Series is either InChIKey or connectivity
                           (depending on whether stereochemistry data are being use or not). The second Series must be the group assignment
                           of each compound.
    :param random_state: seed to use for train/test splitting and KFold shuffling
    :param verbose: log details to stdout
    :return: both:
                    - a dataframe of the cross-validation results where each line is a fold of PCM modelling
                    - the model fitted on all folds for further use
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
        # active.drop(columns=[endpoint], inplace=True)
        inactive = data[data['Activity_class'].isna() & (data[endpoint] <= activity_threshold)]
        inactive = inactive[~inactive['relation'].str.contains('>')][features_to_ignore]
        inactive.loc[:, 'Activity_class'] = 'N'
        # inactive.drop(columns=[endpoint], inplace=True)
        data = pd.concat([preserved, active, inactive])
        # Change endpoint
        endpoint = 'Activity_class'
        del preserved, active, inactive
    # Get and merge molecular descriptors
    mol_descs = get_molecular_descriptors('connectivity' not in data.columns, mol_descriptors, mol_descriptor_path,
                                          mol_descriptor_chunksize)
    mol_descs = filter_molecular_descriptors(mol_descs, merge_on, data[merge_on].unique())
    data = data.merge(mol_descs, on=merge_on)
    data = data.drop(columns=[merge_on])
    # Get and merge protein descriptors
    prot_descs = get_protein_descriptors(prot_descriptors,
                                         prot_sequences_path if isinstance(prot_descriptors, (Descriptor, Transform))
                                         else prot_descriptor_path,
                                         prot_descriptor_chunksize,
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
    # Scale data and reorganize
    scaler = StandardScaler()
    X_train, y_train = training_set.drop(columns=[endpoint]), training_set[[endpoint]]
    X_train.loc[X_train.index, X_train.columns] = scaler.fit_transform(X_train)
    X_test, y_test = test_set.drop(columns=[endpoint]), test_set[[endpoint]]
    X_test.loc[X_test.index, X_test.columns] = scaler.transform(X_test)
    training_set = pd.concat([y_train, X_train], axis=1)
    test_set = pd.concat([y_test, X_test], axis=1)
    del X_train, y_train, X_test, y_test
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
    performance, model = crossvalidate_model(training_set, model, kfold, training_groups, verbose=True)
    X_test, y_test = test_set.iloc[:, 1:], test_set.iloc[:, 0].values.ravel()
    performance.loc['Test set'] = model_metrics(model, y_test, X_test)
    if isinstance(model, (xgboost.XGBRegressor, xgboost.XGBClassifier)):
        warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    return performance, model

def dnn(data: pd.DataFrame,
        endpoint: str = 'pchembl_value_Mean',
        pcm: bool = False,
        num_points: int = 30,
        delta_activity: float = 2,
        mol_descriptors: str = 'mold2',
        mol_descriptor_path: str = './',
        mol_descriptor_chunksize: Optional[int] = 50000,
        prot_sequences_path: str = './',
        prot_descriptors: Union[str, Descriptor, Transform] = 'unirep',
        prot_descriptor_path: str = './',
        prot_descriptor_chunksize: Optional[int] = 50000,
        activity_threshold: float = 6.5,
        model: Optional[BaseNN] = None,
        folds: int = 5,
        stratify: bool = False,
        split_by: str = 'Year',
        split_year: int = 2013,
        validation_set_size: float = 0.20,
        test_set_size: float = 0.30,
        cluster_method: ClusterMixin = None,
        custom_groups: pd.DataFrame = None,
        scale: bool = True,
        random_state: int = 1234,
        verbose: bool = True
        ) -> Tuple[pd.DataFrame, Union[RegressorMixin, ClassifierMixin]]:
    """Create PCM models for as many targets with selected data source(s),
    data quality, minimum number of datapoints and minimum activity amplitude.

    :param data: Papyrus activity data
    :param endpoint: value to be predicted or to derive classes from
    :param pcm: should the DNN model be PCM model, otherwise QSAR
    :param num_points: minimum number of points for the activity of a target to be modelled
    :param delta_activity: minimum difference between most and least active compounds for a target to be modelled
    :param mol_descriptors: type of desriptors to be used for model training
    :param mol_descriptor_path: path to Papyrus descriptors
    :param mol_descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded (None disables chunking)
    :param prot_sequences_path: path to Papyrus sequences
    :param prot_descriptors: type of desriptors to be used for model training
    :param prot_descriptor_path: path to Papyrus descriptors
    :param prot_descriptor_chunksize: chunk size of molecular descriptors to be iteratively loaded (None disables chunking)
    :param activity_threshold: threshold activity between acvtive and inactive compounds (ignored if using a regressor)
    :param model: DNN model to be fitted (default: None = SingleTaskNNClassifier
    :param folds: number of cross-validation folds to be performed
    :param stratify: whether to stratify folds for cross validation, ignored if model is RegressorMixin
    :param split_by: how should folds be determined {'random', 'Year', 'cluster', 'custom'}
                      If 'random', exactly test_set_size is extracted for test set.
                      If 'Year', the size of the test and training set are not looked at
                      If 'cluster' or 'custom', the groups giving proportion closest to test_set_size will be used to defined the test set
    :param split_year: Year from which on the test set is extracted (ignored if split_by is not 'Year')
    :param test_set_size: proportion of the dataset to be used as test set
    :param cluster_method: clustering method to use to extract test set and cross-validation folds (ignored if split_by is not 'cluster')
    :param custom_groups: custom groups to use to extract test set and cross-validation fold (ignored if split_by is not 'custom').
                           Groups must be a pandas DataFrame with only two Series. The first Series is either InChIKey or connectivity
                           (depending on whether stereochemistry data are being use or not). The second Series must be the group assignment
                           of each compound.
    :param scale: should to data be scaled to zero mean and unit variance
    :param random_state: seed to use for train/test splitting and KFold shuffling
    :param verbose: log details to stdout
    :return: both:
                    - a dataframe of the cross-validation results where each line is a fold of PCM modelling
                    - the model fitted on all folds for further use
    """
    if split_by.lower() not in ['year', 'random', 'cluster', 'custom']:
        raise ValueError("split not supported, must be one of {'Year', 'random', 'cluster', 'custom'}")
    if not isinstance(model, (RegressorMixin, ClassifierMixin, SingleTaskNNClassifier, SingleTaskNNRegressor, MultiTaskNNClassifier, MultiTaskNNRegressor)):
        raise ValueError('model type can only be a Scikit-Learn compliant regressor or classifier')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if model is None:
        model = SingleTaskNNClassifier('./')
    model_type = 'regressor' if isinstance(model, (SingleTaskNNRegressor, MultiTaskNNRegressor)) else 'classifier'
    # Keep only required fields
    merge_on = 'connectivity' if 'connectivity' in data.columns else 'InChIKey'
    if model_type == 'regressor':
        features_to_ignore = [merge_on, 'target_id', endpoint, 'Year']
        data = data[data['relation'] == '='][features_to_ignore]
        old_endpoint = []
    else:
        features_to_ignore = [merge_on, 'target_id', 'Activity_class', 'Year']
        preserved = data[~data['Activity_class'].isna()]
        preserved = preserved.drop(
            columns=[col for col in preserved if col not in [merge_on, 'target_id', 'Activity_class', 'Year']])
        active = data[data['Activity_class'].isna() & (data[endpoint] > activity_threshold)]
        active = active[~active['relation'].str.contains('<')][features_to_ignore]
        active.loc[:, 'Activity_class'] = 'A'
        # active.drop(columns=[endpoint], inplace=True)
        inactive = data[data['Activity_class'].isna() & (data[endpoint] <= activity_threshold)]
        inactive = inactive[~inactive['relation'].str.contains('>')][features_to_ignore]
        inactive.loc[:, 'Activity_class'] = 'N'
        # inactive.drop(columns=[endpoint], inplace=True)
        data = pd.concat([preserved, active, inactive])
        # Change endpoint
        endpoint = 'Activity_class'
        del preserved, active, inactive
    # Get and merge molecular descriptors
    mol_descs = get_molecular_descriptors('connectivity' not in data.columns, mol_descriptors, mol_descriptor_path,
                                          mol_descriptor_chunksize)
    mol_descs = filter_molecular_descriptors(mol_descs, merge_on, data[merge_on].unique())
    data = data.merge(mol_descs, on=merge_on)
    data = data.drop(columns=[merge_on])
    if pcm:
        # Get and merge protein descriptors
        prot_descs = get_protein_descriptors(prot_descriptors,
                                             prot_sequences_path if isinstance(prot_descriptors, (Descriptor, Transform))
                                             else prot_descriptor_path,
                                             prot_descriptor_chunksize,
                                             data['target_id'].unique())
        data = data.merge(prot_descs, on='target_id')
        del prot_descs
    # Transform for multi-task model
    if isinstance(model, (MultiTaskNNRegressor, MultiTaskNNClassifier)):
        targets = data['target_id'].unique()
        data.loc[:, targets] = np.zeros((data.shape, len(targets)))
        for target in targets:
            mask = np.where(data.target_id == target)
            data.loc[mask, target] = data.loc[mask, endpoint]
        data = data.drop(columns=[endpoint])
        endpoint = targets
    data = data.drop(columns=['target_id'])
    # Build model for targets reaching criteria
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
        training_groups = training_set['Year']
    elif split_by.lower() == 'random':
        training_groups = None
        training_set, test_set = train_test_split(data,
                                                  test_size=test_set_size,
                                                  random_state=random_state)
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
    training_set, validation_set = train_test_split(training_set,
                                                    test_size=validation_set_size,
                                                    random_state=random_state)
    # Drop columns not used for training
    training_set = training_set.drop(columns=['Year'])
    validation_set = validation_set.drop(columns=['Year'])
    test_set = test_set.drop(columns=['Year'])
    # Scale data and reorganize
    X_train, y_train = training_set.drop(columns=[endpoint]), training_set[[endpoint]]
    X_validation, y_validation = validation_set.drop(columns=[endpoint]), validation_set[[endpoint]]
    X_test, y_test = test_set.drop(columns=[endpoint]), test_set[[endpoint]]
    if scale:
        scaler = StandardScaler()
        X_train.loc[X_train.index, X_train.columns] = scaler.fit_transform(X_train)
        X_validation.loc[X_validation.index, X_validation.columns] = scaler.transform(X_validation)
        X_test.loc[X_test.index, X_test.columns] = scaler.transform(X_test)
    # Make sure enough data
    if model_type == 'classifier':
        enough_data = np.all(np.array(list(Counter(training_set['Activity_class']).values())) > folds)
        if not enough_data:
            print(Counter(training_set['Activity_class']))
            raise ValueError(f'Too few data points for some classes: expected at least {folds} in total')
    # Encode labels if not integers
    lblenc = LabelEncoder()
    y_train = lblenc.fit_transform(y_train.values.ravel())
    y_test = lblenc.transform(y_test.values.ravel())
    y_validation = lblenc.transform(y_validation.values.ravel())
    y_test = y_test.astype(np.int32)
    y_validation = y_validation.astype(np.int32)
    # Combine sets
    training_set = pd.concat([pd.Series(y_train), X_train], axis=1)
    test_set = pd.concat([pd.Series(y_test), X_test], axis=1)
    del y_train, X_test, y_test
    # Define folding scheme for cross validation
    if stratify and model_type == 'classifier':
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    else:
        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    # Set validation set
    model.set_validation(X_validation, y_validation)
    # Set architecture
    if isinstance(model, SingleTaskNNRegressor):
        model.set_architecture(X_train.shape[1])
    elif isinstance(model, SingleTaskNNClassifier):
        model.set_architecture(X_train.shape[1], 1)
    else:
        model.set_architecture(X_train.shape[1], len(endpoint))
    del X_train
    performance, model = crossvalidate_model(training_set, model, kfold, verbose=True)
    X_test, y_test = test_set.iloc[:, 1:], test_set.iloc[:, 0].values.ravel()
    performance.loc['Test set'] = model_metrics(model, y_test, X_test)
    warnings.filterwarnings("default", category=RuntimeWarning)
    return performance, model