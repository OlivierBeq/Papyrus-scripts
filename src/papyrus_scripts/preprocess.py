# -*- coding: utf-8 -*-

"""Filtering capacities of the Papyrus-scripts."""

import os
from itertools import chain
from typing import Any, List, Optional, Union, Iterator, Iterable

import numpy as np
import pandas as pd
import swifter
from joblib import Parallel, delayed
from pandas.io.parsers import TextFileReader as PandasTextFileReader
from sklearn.utils import shuffle
from scipy.stats import median_abs_deviation as MAD
from tqdm.auto import tqdm

from .fingerprint import Fingerprint, MorganFingerprint
from .subsim_search import FPSubSim2


def equalize_cell_size_in_row(row, cols=None, fill_mode='internal', fill_value: object = ''):
    """Equalize the number of values in each list-containing cell of a pandas dataframe.
    
Slightly adapted from user nphaibk:
https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe
    
    :param row: pandas row the function should be applied to
    :param cols: columns for which equalization must be performed
    :param fill_mode: 'internal' to repeat the only/last value of a cell as much as needed
                      'external' to repeat fill_value as much as needed
                      'trim' to remove unaligned values
    :param fill_value: value to repeat as much as needed to equalize cells
    :return: the row with each cell having the same number of values
    """
    if not cols:
        cols = row.index
    # Obtain indices of columns of interest
    jcols = [j for j, v in enumerate(row.index) if v in cols]
    if len(jcols) < 1:
        jcols = range(len(row.index))
    # Obtain lengths of each values
    Ls = [len(x) for x in row.values]
    # Ensure not all values are the same
    if not Ls[:-1] == Ls[1:]:
        # Ensure values are lists
        vals = [v if isinstance(v, list) else [v] for v in row.values]
        if fill_mode == 'external':
            vals = [[e] + [fill_value] * (max(Ls) - 1) if (not j in jcols) and (isinstance(row.values[j], list))
                    else e + [fill_value] * (max(Ls) - len(e))
                    for j, e in enumerate(vals)]
        elif fill_mode == 'internal':
            vals = [[e] + [e] * (max(Ls) - 1) if (not j in jcols) and (isinstance(row.values[j], list))
                    else e + [e[-1]] * (max(Ls) - len(e))
                    for j, e in enumerate(vals)]
        elif fill_mode == 'trim':
            vals = [e[0:min(Ls)] for e in vals]
        else:
            raise ValueError("fill_mode must be one of ['internal', 'external', 'trim']")
        row = pd.Series(vals, index=row.index.tolist())
    return row


def keep_quality(data: Union[pd.DataFrame, PandasTextFileReader, Iterator],
                 min_quality: str = 'high') -> Union[pd.DataFrame, Iterator]:
    """Keep only the data with the minimum defined quality
    
    :param data: the dataframe, chunked or not into a pandas TextFileReader, containing data to be filtered
                 or an Iterator of data chunks
    :param min_quality: minimal quality {'high', 'medium', 'low'} to be kept
                        e.g. if 'medium', data of 'medium' and 'high' quality are kept
    :return: the data with minimal required quality.
             If input is a TextFileReader or an Iterator, the return type is an Iterator
    """
    qualities = ["low", "medium", "high"]
    if min_quality.lower() not in qualities:
        raise ValueError(f'Quality not supported, must be one of {qualities}')
    index = qualities.index(min_quality.lower())
    if isinstance(data, pd.DataFrame):
        filtered = data[data['Quality'].str.lower().isin(qualities[index:])]
        return filtered
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_quality(data, min_quality)
    raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')


def _chunked_keep_quality(chunks: Union[PandasTextFileReader, Iterator], min_quality: str = 'high'):
    for chunk in chunks:
        filtered_chunk = keep_quality(chunk, min_quality)
        yield filtered_chunk


def process_group(group, additional_columns: Optional[List[str]] = None):
    """Aggregate data from one group accordingly"""
    if (group.values[0] == group.values).all():  # If all values are equal, return first record
        group['pchembl_value_Mean'] = group['pchembl_value']
        group['pchembl_value_StdDev'] = np.NaN
        group['pchembl_value_SEM'] = np.NaN
        group['pchembl_value_N'] = 1
        group['pchembl_value_Median'] = group['pchembl_value']
        group['pchembl_value_MAD'] = np.NaN
        return group.iloc[:1, :]
    # Lambda: Return one value if all are the same
    listvals = lambda x: ';'.join(set(str(y) for y in x)) if (x.values[0] == x.values).all() else ';'.join(
        str(y) for y in x)
    # Lambda: Return all values everytime
    listallvals = lambda x: ';'.join(str(y) for y in x)
    # Aggregation rules
    mappings = {'source': 'first', 'CID': listvals, 'AID': listvals,
                'type_IC50': listvals, 'type_EC50': listvals, 'type_KD': listvals,
                'type_Ki': listvals, 'type_other': listvals, 'relation': listvals,
                'pchembl_value': listallvals}
    # Consider other columns
    if additional_columns is not None:
        for column in additional_columns:
            mappings[column] = listvals
    return pd.concat([group.groupby('Activity_ID').aggregate(mappings).reset_index(),
                      group.groupby('Activity_ID')['pchembl_value'].aggregate(pchembl_value_Mean='mean',
                                                                              pchembl_value_StdDev='std',
                                                                              pchembl_value_SEM='sem',
                                                                              pchembl_value_N='count',
                                                                              pchembl_value_Median='median',
                                                                              pchembl_value_MAD=MAD
                                                                              ).reset_index(drop=True)], axis=1)


def process_groups(groups, additional_columns: Optional[List[str]] = None):
    """Aggregate data from multiple groups"""
    return pd.concat([process_group(group, additional_columns) for group in groups])


def keep_source(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], source: Union[List[str], str] = 'all',
                njobs: int = 1, verbose: bool = False) -> pd.DataFrame:
    """Keep only the data from the defined source(s).
    
    :param data: the dataframe containing data to be filtered
    :param source: source(s) to be kept, 'all' or ''any' to keep all data
    :param njobs: number of cores on which multiple processes are spawned to speed up filtering
    :param verbose: whether to show progress bars
    :return: the data with only from the specified source(s),;
             aggregated data (mean, meadians, SEM, ...) are re-calculated to match only
             the specified source(s)
    """
    # Deal with chunked data
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_source(data, source, njobs)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    # Get sources of dataset
    sources_ = set(chain.from_iterable(map(lambda x: x.split(';'), data['source'].unique())))
    sources = set(map(str.lower, sources_))
    # Change type of source if str
    if isinstance(source, str):
        source = [source]
    source = list(map(str.lower, source))
    # Keep all data if source is a list containing 'any', 'all' or all accepted values
    if 'any' in source or 'all' in source or len(set(source).intersection(sources)) == len(sources):
        return data
    # Source not defined
    elif set(source).difference(sources):
        raise ValueError(f'Source not supported, must be one of {sources}')
    # Sources are defined
    else:
        # Columns with optional multiple values
        cols2split = ['source', 'CID', 'AID', 'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other', 'relation',
                      'pchembl_value']
        # Allow processing of Papyrus++
        papyruspp = 'Activity_class' not in data.columns
        if papyruspp:
            data['Activity_class'] = np.NaN
            data['type_other'] = np.NaN
        # Keep trace of order of columns
        ordered_columns = data.columns.tolist()
        # Keep binary data associated to source
        preserved_binary = data[~data['Activity_class'].isna() & data['source'].str.lower().isin(source)]
        # Separate data with multiple sources
        binary_data = data[
            ~data['Activity_class'].isna() & data['source'].str.contains(';') & data['source'].str.contains(
                '|'.join(source), case=False)]
        data = data[data['Activity_class'].isna()]
        if not binary_data.empty:
            # Keep columns and index
            binary_included = binary_data[[x for x in binary_data.columns if x in cols2split + ['Activity_ID']]]
            binary_excluded = binary_data[
                [x for x in binary_data.columns if x not in cols2split and not x.startswith('pchembl_value_')]]
            del binary_data
            binary_included = (
                binary_included.set_index('Activity_ID')  # Alows unnesting data without messing with Activity_ID
                .swifter.progress_bar(verbose)  # Uses swifter without progress bar for apply
                .apply(lambda x: x.str.split(';'))  # Split mutiple values into lists
                .swifter.progress_bar(verbose)
                .apply(equalize_cell_size_in_row, axis=1)  # Set same length of lists in each row
                .swifter.progress_bar(verbose)
                .apply(pd.Series.explode)  # Unnest the data
                .reset_index())  # Recover Activity_ID
            # Filter by sources
            binary_included = binary_included[binary_included['source'].str.lower().isin(source)]
            # Join back with remove columns
            binary_data = binary_included.merge(binary_excluded, how='inner', on='Activity_ID')[ordered_columns]
            del binary_included, binary_excluded
        # Separate records not needing any processing
        preserved = data[data['source'].str.lower().isin(source)]
        # Remove records with non-matching non-unique source
        data = data[
            ~data['source'].str.lower().isin(source) & data['source'].str.contains(';') & data['source'].str.contains(
                '|'.join(source), case=False)]
        if not data.empty:
            # Keep columns and index
            included = data[[x for x in data.columns if x in cols2split + ['Activity_ID']]]
            excluded = data[[x for x in data.columns if x not in cols2split and not x.startswith('pchembl_value_')]]
            del data
            included = (included.set_index('Activity_ID')  # Alows unnesting data without messing with Activity_ID
                        .swifter.progress_bar(verbose)  # Uses swifter without progress bar for apply
                        .apply(lambda x: x.str.split(';'))  # Split mutiple values into lists
                        .swifter.progress_bar(verbose)
                        .apply(equalize_cell_size_in_row, axis=1)  # Set same length of lists in each row
                        .swifter.progress_bar(verbose)
                        .apply(pd.Series.explode)  # Unnest the data
                        .reset_index())  # Recover Activity_ID
            # Filter by sources
            included = included[included['source'].str.lower().isin(source)]
            # Aggregate data on Activity_ID
            _, grouped = list(zip(*included.swifter.progress_bar(verbose).apply(pd.to_numeric, errors='ignore').groupby(
                'Activity_ID')))
            del included
            # Use joblib to speed up the aggregation process
            filtered = pd.concat(Parallel(n_jobs=njobs, backend='loky', verbose=int(verbose))(
                delayed(process_groups)(grouped[i:i + 1000]) for i in range(0, len(grouped), 1000))).reset_index(
                drop=True)
            del grouped
            # Join back with remove columns
            data = filtered.fillna(0).merge(excluded, how='inner', on='Activity_ID')[ordered_columns]
            del excluded, filtered
        # Add back binary data (might be empty)
        data = pd.concat([preserved, data, preserved_binary, binary_data])
        del preserved, preserved_binary, binary_data
        if papyruspp:
            data.drop(columns=['Activity_class', 'type_other'], inplace=True)
        return data


def _chunked_keep_source(data: Union[PandasTextFileReader, Iterator], source: Union[List[str], str],
                         njobs: int) -> pd.DataFrame:
    for chunk in data:
        yield keep_source(chunk, source, njobs)


def is_activity_type(row, activity_types: List[str]):
    """Check if the row matches one of the activity types
    
    :param row: pandas row the function should be applied to
    :param activity_types: activity types the row should partially match
    """
    return np.any([str(row[activity_type]) == '1' for activity_type in activity_types]) and np.all(
        [';' not in str(row[activity_type]) for activity_type in activity_types])


def is_multiple_types(row, activity_types: List[str]):
    """Check if the row matches one of the activity types and if they contain multiple values
    
    :param row: pandas row the function should be applied to
    :param activity_types: activity types with multiple values the row should partially match
    """
    return np.any([';' in str(row[activity_type]) for activity_type in activity_types])


def keep_type(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], activity_types: Union[List[str], str] = 'ic50',
              njobs: int = 1, verbose: bool = False):
    """Keep only the data matching desired activity types
    
    :param data: the dataframe containing data to be filtered
    :param activity_types: type of activity to keep: {'IC50', 'EC50', 'KD', 'Ki', 'all'}
    :param njobs: number of cores on which multiple processes are spawned to speed up filtering
    :param verbose: whether to show progress bars
    :return: the data with desired activity type(s)
    """
    # Deal with chunked data
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_type(data, activity_types, njobs)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    # Define accepted data types
    types = ['IC50', 'EC50', 'KD', 'Ki', 'other']
    types_ = [x.lower() for x in types]
    if isinstance(activity_types, str):
        activity_types = [activity_types]
    activity_types = set([x.lower() for x in activity_types])
    # Keep all data if type is a list containing 'any', 'all' or all accepted values
    if 'any' in activity_types or 'all' in activity_types or len(activity_types.intersection(types_)) == len(types_):
        return data
    # Type not defined
    elif activity_types.difference(types_):
        raise ValueError(f'Type not supported, must be one of {types}')
    else:
        # Allow processing of Papyrus++
        papyruspp = 'Activity_class' not in data.columns
        if papyruspp:
            data['Activity_class'] = np.NaN
            data['type_other'] = np.NaN
        # Transform activity_types to column names
        activity_types = [f"type_{types[i]}" for i in range(len(types)) if types_[i] in activity_types]
        # Columns with optional multiple values
        cols2split = ['source', 'CID', 'AID', 'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other', 'relation',
                      'pchembl_value']
        # Keep trace of order of columns
        ordered_columns = data.columns.tolist()
        # Keep binary data associated to type
        preserved_binary = data[
            ~data['Activity_class'].isna() & data.apply(is_activity_type, activity_types=activity_types, axis=1)]
        # Separate data with multiple types
        binary_data = data[
            ~data['Activity_class'].isna() & data.apply(is_multiple_types, activity_types=activity_types, axis=1)]
        data = data[data['Activity_class'].isna()]
        if not binary_data.empty:
            # Keep columns and index
            binary_included = binary_data[[x for x in binary_data.columns if x in cols2split + ['Activity_ID']]]
            binary_excluded = binary_data[
                [x for x in binary_data.columns if x not in cols2split and not x.startswith('pchembl_value_')]]
            del binary_data
            binary_included = (
                binary_included.set_index('Activity_ID')  # Allows unnesting data without messing with Activity_ID
                .astype(str)  # Required for following split
                .swifter.progress_bar(verbose)  # Uses swifter without progress bar for apply
                .apply(lambda x: x.str.split(';'))  # Split multiple values into lists
                .swifter.progress_bar(verbose)
                .apply(equalize_cell_size_in_row, axis=1)  # Set same length of lists in each row
                .swifter.progress_bar(verbose)
                .apply(pd.Series.explode)  # Unnest the data
                .reset_index())  # Recover Activity_ID
            # Filter by type
            binary_included = binary_included[
                binary_included.swifter.progress_bar(verbose).apply(is_activity_type, activity_types=activity_types,
                                                                    axis=1)]
            # Join back with remove columns
            binary_data = binary_included.merge(binary_excluded, how='inner', on='Activity_ID')[ordered_columns]
            del binary_included, binary_excluded
        # Separate records not needing any processing
        preserved = data[data.apply(is_activity_type, activity_types=activity_types, axis=1)]
        # Remove records with non-matching non-unique type
        data = data[data.apply(is_multiple_types, activity_types=activity_types, axis=1)]
        if not data.empty:
            # Keep columns and index
            included = data[[x for x in data.columns if x in cols2split + ['Activity_ID']]]
            excluded = data[[x for x in data.columns if x not in cols2split and not x.startswith('pchembl_value_')]]
            del data
            included = (included.set_index('Activity_ID')  # Alows unnesting data without messing with Activity_ID
                        .astype(str)  # Required for following split
                        .swifter.progress_bar(verbose)  # Uses swifter without progress bar for apply
                        .apply(lambda x: x.str.split(';'))  # Split multiple values into lists
                        .swifter.progress_bar(verbose)
                        .apply(equalize_cell_size_in_row, axis=1)  # Set same length of lists in each row
                        .swifter.progress_bar(verbose)
                        .apply(pd.Series.explode)  # Unnest the data
                        .reset_index())  # Recover Activity_ID
            # Filter by types
            included = included[included.apply(is_activity_type, activity_types=activity_types, axis=1)]
            # Aggregate data on Activity_ID
            _, grouped = list(zip(*included.swifter.progress_bar(verbose).apply(pd.to_numeric, errors='ignore').groupby(
                'Activity_ID')))
            del included
            # Use joblib to speed up the aggregation process
            filtered = pd.concat(Parallel(n_jobs=njobs, backend='loky', verbose=int(verbose))(
                delayed(process_groups)(grouped[i:i + 1000]) for i in range(0, len(grouped), 1000))).reset_index(
                drop=True)
            del grouped
            # Join back with remove columns
            data = filtered.fillna(0).merge(excluded, how='inner', on='Activity_ID')[ordered_columns]
            del excluded, filtered
        # Add back binary data (might be empty)
        data = pd.concat([preserved, data, preserved_binary, binary_data])
        del preserved, preserved_binary, binary_data
        if papyruspp:
            data.drop(columns=['Activity_class', 'type_other'], inplace=True)
        return data


def _chunked_keep_type(data: Union[PandasTextFileReader, Iterator], activity_types: Union[List[str], str], njobs: int):
    for chunk in data:
        yield keep_type(chunk, activity_types, njobs)


def keep_accession(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], accession: Union[List[str], str] = 'all'):
    """Keep only the data matching desired accession.

    :param data: the dataframe containing data to be filtered
    :param accession: accession to keep (e.g. 'P30542'); mutation can be specified (e.g. '')
    :return: the data with desired accession(s)
    """
    # Deal with chunked data
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_accession(data, accession)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    if isinstance(accession, str):
        accession = [accession]
    return data[data['target_id'].str.lower().str.contains('|'.join(accession).lower())]


def _chunked_keep_accession(data: Union[PandasTextFileReader, Iterator], accession: Union[List[str], str]):
    for chunk in data:
        filtered_chunk = keep_accession(chunk, accession)
        yield filtered_chunk


def equalize_cell_size_in_column(col, fill_mode='internal', fill_value: object = ''):
    """Equalize the number of values in each list-containing cell of a pandas dataframe.

    Adapted from user nphaibk
https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe

    :param col: pandas Series the function should be applied to
    :param fill_mode: 'internal' to repeat the only/last value of a cell as much as needed
                      'external' to repeat fill_value as much as needed
                      'trim' to remove unaligned values
    :param fill_value: value to repeat as much as needed to equalize cells
    :return: the column with each cell having the same number of values
    """
    Ls = [len(x) for x in col.values]
    if not Ls[:-1] == Ls[1:]:
        vals = [v if isinstance(v, list) else [v] for v in col.values]
        if fill_mode == 'external':
            vals = [e + [fill_value] * (max(Ls) - len(e)) for j, e in enumerate(vals)]
        elif fill_mode == 'internal':
            vals = [e + [e[-1]] * (max(Ls) - len(e)) for j, e in enumerate(vals)]
        elif fill_mode == 'trim':
            vals = [e[0:min(Ls)] for e in vals]
        else:
            raise ValueError("fill_mode must be one of ['internal', 'external', 'trim']")
        col = pd.Series(vals, index=col.index.tolist())
    return col


def keep_protein_class(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], protein_data: pd.DataFrame,
                       classes: Optional[Union[dict, List[dict]]] = [{'l2': 'Kinase'}, {'l5': 'Adenosine receptor'}],
                       generic_regex: bool = False):
    """Keep only the data matching desired protein classifications.

    :param data: the dataframe containing data to be filtered
    :param protein_data: the dataframe of Papyrus protein targets
    :param classes: protein classes to keep (case insensitive).
    - {'l2': 'Kinase'} matches all proteins with classification 'Enzyme->Kinase'
    - {'l5': 'Adenosine receptor'} matches 'Membrane receptor->Family A G protein-coupled receptor->Small molecule receptor (family A GPCR)->Nucleotide-like receptor (family A GPCR)-> Adenosine receptor'
    - All levels in the same dict are enforced, e.g. {'l1': ''Epigenetic regulator', 'l3': 'HDAC class IIb'} does not match records without the specified l1 AND l3
    - If given a list of dicts, results in a union of the dicts, e.g. [{'l2': 'Kinase'}, {'l1': 'Membrane receptor'}] matches records with classification either 'Enzyme->Kinase' or 'Membrane receptor'
    - Level-independent patterns can be specified with the 'l?' key, e.g. {'l?': 'SLC'} matches any classification level containing the 'SLC' keyword
    Only one 'l?' per dict is supported.
    Mixed usage of 'l?' and level-specific patterns (e.f. 'l1') is not supported
    :param generic_regex: whether to consider generic patterns 'l?' as regex, allowing for partial match.

    :return: the data with desired protein classes
    """
    # Deal with chunked data
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_protein_class(data, protein_data, classes, generic_regex)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    # If no filter return entire dataset
    if classes is None:
        return data
    if isinstance(classes, dict):
        classes = [classes]
    # Verify classification keys
    keys = set(key for keys in classes for key in keys.keys())
    allowed_keys = ['l?', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8']
    if keys.difference(allowed_keys):
        raise ValueError(f'levels of protein classes must be of {allowed_keys}')
    lvl_dependent, lvl_independent = False, False
    for key in classes:
        if 'l?' in key.keys():
            lvl_independent = True
            if len(key.keys()) > 1:
                raise ValueError(f'only one pattern per "l?" is accepted')
        else:
            lvl_dependent = True
    # Split classifications
    ## 1) Handle multiple classifications
    split_classes = protein_data['Classification'].str.split(';')
    split_classes = equalize_cell_size_in_column(split_classes, 'external', '')
    split_classes = pd.DataFrame(split_classes.tolist())
    ## 2) Split into classification levels
    multiplicity = len(split_classes.columns)  # Number of max classifications
    for j in range(multiplicity):
        split_classes.iloc[:, j] = split_classes.iloc[:, j].str.split('->')
        split_classes.iloc[:, j] = equalize_cell_size_in_column(split_classes.iloc[:, j], 'external', '')
        # Ensure 8 levels of classification
        for _ in range(8 - len(split_classes.iloc[0, j])):
            split_classes.iloc[0, j].append('')
        split_classes.iloc[:, j] = equalize_cell_size_in_column(split_classes.iloc[:, j])
    ## 3) Create DataFrame with all annotations
    split_classes = pd.concat(
        [pd.DataFrame(split_classes.iloc[:, j].tolist(), columns=[f'l{x + 1}_{j + 1}' for x in range(8)]) for j in
         range(multiplicity)], axis=1)
    # Ensure case insensitivity
    split_classes = split_classes.apply(lambda s: s.str.lower())
    # Filter classes
    ## 1) Deal with specific protein classes (i.e. l1 to l8)
    if lvl_dependent:
        query_dpd = ') or ('.join([') or ('.join([' and '.join([f'`{subkey.lower()}_{k + 1}` == "{subval.lower()}"'
                                                                for subkey, subval in key.items()
                                                                ])
                                                  for k in range(multiplicity)
                                                  ])
                                   for key in classes if 'l?' not in key.keys()
                                   ])
    ## 2) Deal with 'l?'
    regex_indices = []
    if lvl_independent:
        query_idpd = ""
        if generic_regex:  # Use regex
            regex_indices = split_classes[
                eval('|'.join([f'split_classes["{subkey.lower()}"].str.lower().str.contains("{subval.lower()}", regex=True)'
                               for key in classes for subkey in split_classes.columns for subval in key.values() if
                               'l?' in key.keys()])
                     )].index.tolist()
        else:  # Complete match
            query_idpd = ') or ('.join([') or ('.join([' and '.join([f'`{subkey.lower()}` == "{subval.lower()}"'
                                                                     for subval in key.values()
                                                                     ])
                                                       for subkey in split_classes.columns
                                                       ])
                                        for key in classes if 'l?' in key.keys()
                                        ])
    query = (f"{('(' + query_dpd + ')') if lvl_dependent else ''}"
             f"{' or ' if lvl_dependent and lvl_independent and not generic_regex else ''}"
             f"{('(' + query_idpd + ')') if lvl_independent and not generic_regex else ''}")
    ##  3) Execute filter
    if len(query):
        indices = split_classes.query(query).index.tolist()
    else:
        indices = []
    if generic_regex:
        indices = sorted(set(indices + regex_indices))
    # Obtain targets from filtered indices
    targets = protein_data.loc[indices, 'target_id']
    # Map back to activity data
    return data[data['target_id'].isin(targets)].merge(protein_data.loc[indices, ('target_id', 'Classification')], on='target_id')


def _chunked_keep_protein_class(data: Union[PandasTextFileReader, Iterator], protein_data: pd.DataFrame,
                                classes: Optional[Union[dict, List[dict]]],
                                generic_regex: bool):
    for chunk in data:
        filtered_chunk = keep_protein_class(chunk, protein_data, classes, generic_regex)
        yield filtered_chunk


def consume_chunks(generator: Union[PandasTextFileReader, Iterator], progress: bool = True, total: int = None):
    """Transform the result of chained filters into a pandas DataFrame

    :param generator: iterator to be transformed into a dataframe
    :param progress: whether to show progress
    :param total: total number of chunks the input is divided in
    """
    data = []
    if progress:
        pbar = tqdm(generator, total=total)
    else:
        pbar = generator
    for item in pbar:
        if not isinstance(item, pd.DataFrame):
            consumed = _consume_deeper_chunks(item)
            data.extend(consumed)
        else:
            data.append(item)
    if not len(data):
        return pd.DataFrame()
    return pd.concat(data, axis=0)


def _consume_deeper_chunks(generator: Union[PandasTextFileReader, Iterator]):
    """Transform the result of chained filters into a pandas DataFrame.

    Internal function. One must use consume_chunks instead.

    :param generator: iterator to be transformed into a dataframe
    """
    data = []
    for item in generator:
        if not isinstance(item, pd.DataFrame):
            consumed = consume_chunks(item)
            data.extend(consumed)
        else:
            data.append(item)
    if not len(data):
        return pd.DataFrame()
    return pd.concat(data, axis=0)


def keep_organism(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], protein_data: pd.DataFrame,
                  organism: Optional[Union[str, List[str]]] = 'Homo sapiens (Human)',
                  generic_regex: bool = False):
    """Keep only the data matching desired protein classifications.

    :param data: the dataframe containing data to be filtered
    :param protein_data: the dataframe of Papyrus protein targets
    :param organism: organisms to keep (case insensitive).
    :param generic_regex: whether to allow for partial match.

    :return: the data with desired protein classes
    """
    # Deal with chunked data
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_organism(data, protein_data, organism, generic_regex)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    # If no filter return entire dataset
    if organism is None:
        return data
    if isinstance(organism, str):
        organism = [organism]
    if generic_regex:  # apply regex
        indices = protein_data[
            eval('|'.join([f'protein_data["Organism"].str.lower().str.contains("{x.lower()}", regex=True)'
                           for x in organism])
                 )].index.tolist()
    else:
        query = '(' + ' or '.join([f'Organism == "{x}"' for x in organism]) + ')'
        indices = protein_data.query(query).index.tolist()
    # Obtain targets from filtered indices
    targets = protein_data.loc[indices, 'target_id']
    # Map back to activity data
    return data[data['target_id'].isin(targets)].merge(protein_data.loc[indices, ('target_id', 'Organism')], on='target_id')


def _chunked_keep_organism(data: Union[PandasTextFileReader, Iterator], protein_data: pd.DataFrame,
                           organism: Optional[Union[str, List[str]]],
                           generic_regex: bool):
    for chunk in data:
        filtered_chunk = keep_organism(chunk, protein_data, organism, generic_regex)
        yield filtered_chunk


def keep_match(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], column: str, values: Union[Any, List[Any]]):
    """Keep only the data matching desired columns with desired values (equivalent to *isin*).

    :param data: the dataframe containing data to be filtered
    :param column: column to be filtered
    :param values: values to be retained

    :return: the data with desired column values
    """
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_match(data, column, values)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    if not isinstance(values, list):
        values = [values]
    return data[data[column].isin(values)]


def _chunked_keep_match(data: Union[PandasTextFileReader, Iterator], column: str, values: Union[Any, List[Any]]):
    for chunk in data:
        filtered_chunk = keep_match(chunk, column, values)
        yield filtered_chunk


def keep_contains(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], column: str, value: str, case: bool = True, regex: bool = False):
    """Keep only the data matching desired columns containing desired values.

    :param data: the dataframe containing data to be filtered
    :param column: column to be filtered
    :param value: value to be retained
    :param case: whether value is case-sensitive
    :param regex: whether to interpret value as a regular expression

    :return: the data containing desired values
    """
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_contains(data, column, value, case, regex)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    return data[data[column].str.contains(value, case=case, regex=regex)]


def _chunked_keep_contains(data: Union[PandasTextFileReader, Iterator], column: str, value: str, case: bool = True, regex: bool = False):
    for chunk in data:
        filtered_chunk = keep_contains(chunk, column, value, case, regex)
        yield filtered_chunk


def keep_similar(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], molecule_smiles: Union[str, List[str]], fpsubsim2_file: str, fingerprint: Fingerprint = MorganFingerprint(), threshold: float = 0.7, cuda: bool = False):
    """Keep only data associated to molecules similar to the query.

    :param data: the dataframe containing data to be filtered
    :param molecule_smiles: the query molecule(s)
    :param fpsubsim2_file: path to FPSubSim2 database
    :param fingerprint: fingerprint to be used for similarity search
    :param threshold: similarity threshold
    :param cuda: whether to use GPU for similarity searches

    :return: the data associated to similar molecules
    """
    if not os.path.isfile(fpsubsim2_file):
        raise ValueError(f'FPSubSim2 database does not exist: {fpsubsim2_file}')
    fpss2 = FPSubSim2()
    fpss2.load(fpsubsim2_file)
    if str(fingerprint) not in fpss2.available_fingerprints.keys():
        raise ValueError(f'FPSubSim2 database does not contain fingerprint {fingerprint.name}')
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_similar(data, molecule_smiles, fpsubsim2_file, fingerprint, threshold, cuda)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    # Obtain similar molecules
    similarity_engine = fpss2.get_similarity_lib(cuda=cuda)
    similar_mols = pd.concat([similarity_engine.similarity(smiles, threshold=threshold) for smiles in tqdm(molecule_smiles)], axis=0)
    similar_mols = similar_mols.iloc[:, -2:]
    filtered_data = data[data['InChIKey'].isin(similar_mols['InChIKey'])].merge(similar_mols, on='InChIKey')
    return filtered_data


def _chunked_keep_similar(data: Union[PandasTextFileReader, Iterator], molecule_smiles: str, fpsubsim2_file: str, fingerprint: Fingerprint, threshold: float = 0.7, cuda: bool = False):
    fpss2 = FPSubSim2()
    fpss2.load(fpsubsim2_file)
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    similarity_engine = fpss2.get_similarity_lib(cuda=cuda, fp_signature=fingerprint._hash)
    similar_mols = pd.concat(
        [similarity_engine.similarity(smiles, threshold=threshold) for smiles in tqdm(molecule_smiles)], axis=0)
    similar_mols = similar_mols.iloc[:, -2:]
    for chunk in data:
        filtered_chunk = chunk[chunk['InChIKey'].isin(similar_mols['InChIKey'])].merge(similar_mols, on='InChIKey')
        yield filtered_chunk


def keep_substructure(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], molecule_smiles: Union[str, List[str]], fpsubsim2_file: str):
    """Keep only data associated to molecular substructures of the query.

    :param data: the dataframe containing data to be filtered
    :param molecule_smiles: the query molecule(s)
    :param fpsubsim2_file: path to FPSubSim2 database

    :return: the data associated to similar molecules
    """
    if not os.path.isfile(fpsubsim2_file):
        raise ValueError(f'FPSubSim2 database does not exist: {fpsubsim2_file}')
    fpss2 = FPSubSim2()
    fpss2.load(fpsubsim2_file)
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_substructure(data, molecule_smiles, fpsubsim2_file)
    # Raise error if not correct type
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('data can only be a pandas DataFrame, TextFileReader or an Iterator')
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    # Obtain similar molecules
    substructure_engine = fpss2.get_substructure_lib()
    substructure_mols = pd.concat([substructure_engine.substructure(smiles) for smiles in tqdm(molecule_smiles)], axis=0)
    filtered_data = data[data['InChIKey'].isin(substructure_mols['InChIKey'])]
    return filtered_data

def _chunked_keep_substructure(data: Union[PandasTextFileReader, Iterator], molecule_smiles: Union[str, List[str]], fpsubsim2_file: str):
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2 = FPSubSim2()
    fpss2.load(fpsubsim2_file)
    # Obtain similar molecules
    substructure_engine = fpss2.get_substructure_lib()
    substructure_mols = pd.concat([substructure_engine.substructure(smiles) for smiles in tqdm(molecule_smiles)], axis=0)
    for chunk in data:
        filtered_chunk = chunk[chunk['InChIKey'].isin(substructure_mols['InChIKey'])]
        yield filtered_chunk


def yscrambling(data: Union[pd.DataFrame, PandasTextFileReader, Iterator], y_var: Union[str, List[str]] = 'pchembl_value_Mean', random_state: int = 1234):
    """Perform y-scrambling on the variable(s) to be predicted.

    :param data: the data containing the variable(s) to be shuffled
    :param y_var: the name(s) of columns which data should be randomized
    :param random_state: random seed used for shuffling
    :return: the input data with specified variable(s) scrambled
    """
    if not isinstance(y_var, (str, list)):
        raise ValueError('y_var must be either a str or a list')
    if not isinstance(y_var, list):
        y_var = [y_var]
    for var in y_var:
        data[var] = shuffle(data[var], random_state=random_state)
    return data
