# -*- coding: utf-8 -*-

"""Filtering functions for the Papyrus dataset."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from functools import wraps
from itertools import chain
from typing import Any, Callable, Generator, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import swifter  # noqa: F401 — registers the .swifter accessor
from joblib import Parallel, delayed
from pandas.io.parsers import TextFileReader as PandasTextFileReader
from scipy.stats import median_abs_deviation as MAD
from sklearn.utils import shuffle
from tqdm.auto import tqdm

from .fingerprint import Fingerprint, MorganFingerprint
from .subsim_search import FPSubSim2

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Any of the three input forms accepted by every public filter.
DataInput = Union[pd.DataFrame, PandasTextFileReader, Iterator]
#: The corresponding output form.
DataOutput = Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]


# ---------------------------------------------------------------------------
# Chunk-dispatch decorator
# ---------------------------------------------------------------------------

def _supports_chunks(fn: Callable) -> Callable:
    """Transparently iterate *fn* over chunked input.

    When the first argument (*data*) is a :class:`~pandas.io.parsers.TextFileReader`
    or any other :class:`~typing.Iterator`, the decorated function returns a
    generator that applies the original function to each chunk.  For a plain
    :class:`~pandas.DataFrame` the function is called directly.

    This eliminates all ``_chunked_*`` private wrappers and the repetitive
    ``isinstance`` guard at the top of every public filter.
    """

    @wraps(fn)
    def wrapper(data: DataInput, *args, **kwargs) -> DataOutput:
        if isinstance(data, (PandasTextFileReader, Iterator)):
            return (fn(chunk, *args, **kwargs) for chunk in data)
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f'data must be a pd.DataFrame, TextFileReader or Iterator, '
                f'got {type(data).__name__!r}.'
            )
        return fn(data, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Papyrus++ column-injection context manager
# ---------------------------------------------------------------------------

@contextmanager
def _papyruspp_columns(data: pd.DataFrame):
    """Temporarily inject columns absent in the Papyrus++ subset.

    Papyrus++ lacks ``Activity_class`` and ``type_other``.  Many filters rely
    on those columns to distinguish binary-class data from continuous data.
    This context manager injects them as ``NaN`` columns when missing and
    removes them on exit — even if an exception is raised — so callers never
    have to clean up manually.

    :param data: the DataFrame to patch in-place
    """
    added: List[str] = []
    for col in ('Activity_class', 'type_other'):
        if col not in data.columns:
            data[col] = np.nan
            added.append(col)
    try:
        yield data
    finally:
        if added:
            data.drop(columns=added, inplace=True, errors='ignore')


# ---------------------------------------------------------------------------
# Cell-size equalisation helpers  (unchanged logic, kept as-is)
# ---------------------------------------------------------------------------

def equalize_cell_size_in_row(row, cols=None, fill_mode: str = 'internal', fill_value: object = ''):
    """Equalise the number of values in each list-containing cell of a row.

    Slightly adapted from user *nphaibk*:
    https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe

    :param row: pandas Series (a DataFrame row) to equalise
    :param cols: columns to equalise; defaults to all columns
    :param fill_mode: ``'internal'`` repeats the last value, ``'external'``
        repeats *fill_value*, ``'trim'`` truncates to the shortest list
    :param fill_value: value used when *fill_mode* is ``'external'``
    :raises ValueError: if *fill_mode* is not recognised
    """
    if not cols:
        cols = row.index
    jcols = [j for j, v in enumerate(row.index) if v in cols]
    if not jcols:
        jcols = range(len(row.index))
    lengths = [len(x) for x in row.values]
    if lengths[:-1] == lengths[1:]:  # all equal → nothing to do
        return row
    vals = [v if isinstance(v, list) else [v] for v in row.values]
    max_len = max(lengths)
    if fill_mode == 'external':
        vals = [
            [e] + [fill_value] * (max_len - 1)
            if (j not in jcols and isinstance(row.values[j], list))
            else e + [fill_value] * (max_len - len(e))
            for j, e in enumerate(vals)
        ]
    elif fill_mode == 'internal':
        vals = [
            [e] + [e] * (max_len - 1)
            if (j not in jcols and isinstance(row.values[j], list))
            else e + [e[-1]] * (max_len - len(e))
            for j, e in enumerate(vals)
        ]
    elif fill_mode == 'trim':
        min_len = min(lengths)
        vals = [e[:min_len] for e in vals]
    else:
        raise ValueError("fill_mode must be one of ['internal', 'external', 'trim']")
    return pd.Series(vals, index=row.index.tolist())


def equalize_cell_size_in_column(col: pd.Series, fill_mode: str = 'internal', fill_value: object = '') -> pd.Series:
    """Equalise the number of values in each list-containing cell of a column.

    Adapted from user *nphaibk*:
    https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe

    :param col: pandas Series whose cells are lists
    :param fill_mode: ``'internal'`` repeats the last value, ``'external'``
        repeats *fill_value*, ``'trim'`` truncates to the shortest list
    :param fill_value: value used when *fill_mode* is ``'external'``
    :raises ValueError: if *fill_mode* is not recognised
    """
    lengths = [len(x) for x in col.values]
    if lengths[:-1] == lengths[1:]:
        return col
    vals = [v if isinstance(v, list) else [v] for v in col.values]
    max_len, min_len = max(lengths), min(lengths)
    if fill_mode == 'external':
        vals = [e + [fill_value] * (max_len - len(e)) for e in vals]
    elif fill_mode == 'internal':
        vals = [e + [e[-1]] * (max_len - len(e)) for e in vals]
    elif fill_mode == 'trim':
        vals = [e[:min_len] for e in vals]
    else:
        raise ValueError("fill_mode must be one of ['internal', 'external', 'trim']")
    return pd.Series(vals, index=col.index.tolist())


# ---------------------------------------------------------------------------
# Activity-aggregation helpers
# ---------------------------------------------------------------------------

def process_group(group: pd.DataFrame, additional_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Aggregate duplicate activity records within one Activity-ID group.

    :param group: a sub-DataFrame sharing the same ``Activity_ID``
    :param additional_columns: extra columns to include in the aggregation
    :returns: a single-row (or small) aggregated DataFrame with derived
        ``pchembl_value_Mean``, ``_StdDev``, ``_SEM``, ``_N``, ``_Median``,
        ``_MAD`` columns
    """
    if (group.values[0] == group.values).all():
        group['pchembl_value_Mean'] = group['pchembl_value']
        group['pchembl_value_StdDev'] = np.nan
        group['pchembl_value_SEM'] = np.nan
        group['pchembl_value_N'] = 1
        group['pchembl_value_Median'] = group['pchembl_value']
        group['pchembl_value_MAD'] = np.nan
        return group.iloc[:1, :]

    listvals = lambda x: ';'.join(set(str(y) for y in x)) if (x.values[0] == x.values).all() else ';'.join(
        str(y) for y in x
        )
    listallvals = lambda x: ';'.join(str(y) for y in x)

    mappings = {
        'source': 'first', 'CID': listvals, 'AID': listvals,
        'type_IC50': listvals, 'type_EC50': listvals, 'type_KD': listvals,
        'type_Ki': listvals, 'type_other': listvals, 'relation': listvals,
        'pchembl_value': listallvals,
    }
    if additional_columns:
        for col in additional_columns:
            mappings[col] = listvals

    return pd.concat([
        group.groupby('Activity_ID').aggregate(mappings).reset_index(),
        group.groupby('Activity_ID')['pchembl_value'].aggregate(
            pchembl_value_Mean='mean',
            pchembl_value_StdDev='std',
            pchembl_value_SEM='sem',
            pchembl_value_N='count',
            pchembl_value_Median='median',
            pchembl_value_MAD=MAD,
        ).reset_index(drop=True),
    ], axis=1
    )


def process_groups(groups, additional_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Aggregate data from multiple Activity-ID groups.

    :param groups: iterable of group DataFrames (from a ``groupby`` result)
    :param additional_columns: forwarded to :func:`process_group`
    """
    return pd.concat([process_group(g, additional_columns) for g in groups])


# ---------------------------------------------------------------------------
# Row-type helpers
# ---------------------------------------------------------------------------

def is_activity_type(row: pd.Series, activity_types: List[str]) -> bool:
    """Return True when *row* matches one of the *activity_types* unambiguously.

    :param row: a DataFrame row
    :param activity_types: column names (e.g. ``'type_IC50'``) to check
    """
    return (
            np.any([str(row[t]) == '1' for t in activity_types])
            and np.all([';' not in str(row[t]) for t in activity_types])
    )


def is_multiple_types(row: pd.Series, activity_types: List[str]) -> bool:
    """Return True when *row* has semicolon-separated values in any of *activity_types*.

    :param row: a DataFrame row
    :param activity_types: column names to inspect for multi-value entries
    """
    return np.any([';' in str(row[t]) for t in activity_types])


# ---------------------------------------------------------------------------
# Multi-source / multi-type splitting helper
# ---------------------------------------------------------------------------

_COLS_TO_SPLIT = [
    'source', 'CID', 'AID',
    'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other',
    'relation', 'pchembl_value',
]


def _unnest_and_filter(
        df: pd.DataFrame,
        keep_mask: pd.Series,
        ordered_columns: List[str],
        njobs: int,
        verbose: bool,
        aggregate: bool = True,
) -> pd.DataFrame:
    """Split semicolon-joined columns, filter rows, and optionally re-aggregate.

    This is the shared implementation used by :func:`keep_source` and
    :func:`keep_type` when records carry multiple values in a single cell.

    :param df: records with semicolon-separated multi-values to process
    :param keep_mask: boolean Series identifying rows to retain after unnesting
    :param ordered_columns: original column order to restore after merging
    :param njobs: parallelism for the aggregation step
    :param verbose: whether to show swifter progress bars
    :param aggregate: when True, re-aggregate on ``Activity_ID`` after filtering;
        set to False for binary-class records that should not be aggregated
    :returns: filtered and optionally re-aggregated DataFrame
    """
    included = df[[c for c in df.columns if c in _COLS_TO_SPLIT + ['Activity_ID']]]
    excluded = df[[c for c in df.columns if c not in _COLS_TO_SPLIT and not c.startswith('pchembl_value_')]]

    included = (
        included.set_index('Activity_ID')
        .astype(str)
        .swifter.progress_bar(verbose).apply(lambda x: x.str.split(';'))
        .swifter.progress_bar(verbose).apply(equalize_cell_size_in_row, axis=1)
        .swifter.progress_bar(verbose).apply(pd.Series.explode)
        .reset_index()
    )
    included = included[keep_mask(included)]

    if not aggregate or included.empty:
        return included.merge(excluded, how='inner', on='Activity_ID')[ordered_columns]

    _, grouped = list(zip(*(
        included.swifter.progress_bar(verbose)
        .apply(pd.to_numeric, errors='ignore')
        .groupby('Activity_ID')
    )
                          )
                      )
    filtered = pd.concat(
        Parallel(n_jobs=njobs, backend='loky', verbose=int(verbose))(
            delayed(process_groups)(grouped[i:i + 1000])
            for i in range(0, len(grouped), 1000)
        )
    ).reset_index(drop=True)

    return filtered.fillna(0).merge(excluded, how='inner', on='Activity_ID')[ordered_columns]


# ---------------------------------------------------------------------------
# Public filters
# ---------------------------------------------------------------------------

@_supports_chunks
def keep_quality(
        data: pd.DataFrame,
        min_quality: str = 'high',
) -> pd.DataFrame:
    """Keep only rows at or above the minimum required quality level.

    :param data: bioactivity DataFrame (chunked or not)
    :param min_quality: lowest quality to retain: ``'low'``, ``'medium'``,
        or ``'high'``
    :raises ValueError: if *min_quality* is not recognised
    """
    qualities = ['low', 'medium', 'high']
    if min_quality.lower() not in qualities:
        raise ValueError(f'min_quality must be one of {qualities}, got {min_quality!r}')
    threshold = qualities.index(min_quality.lower())
    return data[data['Quality'].str.lower().isin(qualities[threshold:])]


@_supports_chunks
def keep_source(
        data: pd.DataFrame,
        source: Union[List[str], str] = 'all',
        njobs: int = 1,
        verbose: bool = False,
) -> pd.DataFrame:
    """Keep only rows from the specified data source(s).

    Aggregated statistics (mean, median, SEM …) are recomputed to reflect
    only the retained sources.

    :param data: bioactivity DataFrame (chunked or not)
    :param source: source label(s) to retain; ``'all'`` or ``'any'`` keeps
        every source
    :param njobs: number of parallel jobs for re-aggregation
    :param verbose: show swifter progress bars
    """
    sources_ = set(chain.from_iterable(s.split(';') for s in data['source'].unique()))
    sources = set(map(str.lower, sources_))

    if isinstance(source, str):
        source = [source]
    source = [s.lower() for s in source]

    if 'any' in source or 'all' in source or set(source) >= sources:
        return data

    pattern = '|'.join(f'^{re.escape(s)}$' for s in source)
    source_adapted = [s for s in sources if re.search(pattern, s)]

    if not source_adapted:
        return data[data['source'] == 'SOURCE UNAVAILABLE']

    ordered_columns = data.columns.tolist()

    with _papyruspp_columns(data):
        # Binary-class records with a single matching source — keep as-is.
        preserved_binary = data[
            data['Activity_class'].notna()
            & data['source'].str.lower().isin(source_adapted)
            ]
        # Binary-class records with multiple sources — must be unnested first.
        multi_binary = data[
            data['Activity_class'].notna()
            & data['source'].str.contains(';')
            & data['source'].str.lower().str.contains('|'.join(map(re.escape, source_adapted)), case=False)
            ]
        # Continuous records.
        data = data[data['Activity_class'].isna()]

    binary_data = pd.DataFrame()
    if not multi_binary.empty:
        binary_data = _unnest_and_filter(
            multi_binary,
            keep_mask=lambda df: df['source'].str.lower().isin(source_adapted),
            ordered_columns=ordered_columns,
            njobs=njobs, verbose=verbose, aggregate=False,
        )

    preserved = data[data['source'].str.lower().isin(source_adapted)]
    multi_cont = data[
        ~data['source'].str.lower().isin(source_adapted)
        & data['source'].str.contains(';')
        & data['source'].str.lower().str.contains('|'.join(map(re.escape, source_adapted)), case=False)
        ]

    filtered = pd.DataFrame()
    if not multi_cont.empty:
        filtered = _unnest_and_filter(
            multi_cont,
            keep_mask=lambda df: df['source'].str.lower().isin(source_adapted),
            ordered_columns=ordered_columns,
            njobs=njobs, verbose=verbose, aggregate=True,
        )

    return pd.concat(
        [preserved, filtered, preserved_binary, binary_data],
        ignore_index=True,
    )


@_supports_chunks
def keep_type(
        data: pd.DataFrame,
        activity_types: Union[List[str], str] = 'ic50',
        njobs: int = 1,
        verbose: bool = False,
) -> pd.DataFrame:
    """Keep only rows matching the desired activity type(s).

    Aggregated statistics are recomputed to reflect only the retained types.

    :param data: bioactivity DataFrame (chunked or not)
    :param activity_types: type(s) to retain: ``'IC50'``, ``'EC50'``,
        ``'KD'``, ``'Ki'``, ``'other'``, ``'all'``, or ``'any'``
    :param njobs: number of parallel jobs for re-aggregation
    :param verbose: show swifter progress bars
    :raises ValueError: if any supplied type is not recognised
    """
    canonical = ['IC50', 'EC50', 'KD', 'Ki', 'other']
    lower_map = {t.lower(): t for t in canonical}

    if isinstance(activity_types, str):
        activity_types = [activity_types]
    activity_types = {t.lower() for t in activity_types}

    if 'any' in activity_types or 'all' in activity_types or activity_types >= set(lower_map):
        return data

    unknown = activity_types - set(lower_map)
    if unknown:
        raise ValueError(f'Unrecognised activity type(s): {unknown}. Must be one of {canonical}')

    type_cols = [f'type_{lower_map[t]}' for t in activity_types]
    ordered_columns = data.columns.tolist()

    with _papyruspp_columns(data):
        preserved_binary = data[
            data['Activity_class'].notna()
            & data.apply(is_activity_type, activity_types=type_cols, axis=1)
            ]
        multi_binary = data[
            data['Activity_class'].notna()
            & data.apply(is_multiple_types, activity_types=type_cols, axis=1)
            ]
        data = data[data['Activity_class'].isna()]

    binary_data = pd.DataFrame()
    if not multi_binary.empty:
        binary_data = _unnest_and_filter(
            multi_binary,
            keep_mask=lambda df: df.apply(is_activity_type, activity_types=type_cols, axis=1),
            ordered_columns=ordered_columns,
            njobs=njobs, verbose=verbose, aggregate=False,
        )

    preserved = data[data.apply(is_activity_type, activity_types=type_cols, axis=1)]
    multi_cont = data[data.apply(is_multiple_types, activity_types=type_cols, axis=1)]

    filtered = pd.DataFrame()
    if not multi_cont.empty:
        filtered = _unnest_and_filter(
            multi_cont,
            keep_mask=lambda df: df.apply(is_activity_type, activity_types=type_cols, axis=1),
            ordered_columns=ordered_columns,
            njobs=njobs, verbose=verbose, aggregate=True,
        )

    return pd.concat(
        [preserved, filtered, preserved_binary, binary_data],
        ignore_index=True,
    )


@_supports_chunks
def keep_accession(
        data: pd.DataFrame,
        accession: Union[List[str], str] = 'all',
) -> pd.DataFrame:
    """Keep only rows whose target ID matches the given UniProt accession(s).

    :param data: bioactivity DataFrame (chunked or not)
    :param accession: accession code(s) to retain, e.g. ``'P30542'``.
        Mutation suffixes are supported (e.g. ``'P30542_V52A'``).
    """
    if isinstance(accession, str):
        accession = [accession]
    pattern = '|'.join(re.escape(a) for a in accession)
    return data[data['target_id'].str.lower().str.contains(pattern.lower(), regex=True)]


@_supports_chunks
def keep_protein_class(
        data: pd.DataFrame,
        protein_data: pd.DataFrame,
        classes: Optional[Union[dict, List[dict]]] = None,
        generic_regex: bool = False,
) -> pd.DataFrame:
    """Keep only rows whose target belongs to the desired protein class(es).

    :param data: bioactivity DataFrame (chunked or not)
    :param protein_data: Papyrus protein-target DataFrame
    :param classes: protein class filter(s).

        Each dict maps a level key (``'l1'``–``'l8'``, or ``'l?'`` for any
        level) to the desired class label.  Multiple keys in one dict are
        ANDed; a list of dicts is ORed.  Examples:

        * ``{'l2': 'Kinase'}`` → all kinases
        * ``[{'l2': 'Kinase'}, {'l1': 'Membrane receptor'}]`` → union
        * ``{'l?': 'SLC'}`` → any level containing *SLC*

        ``None`` returns *data* unfiltered.
    :param generic_regex: when True, ``'l?'`` patterns are matched as
        regular expressions (partial match); when False, exact match only
    :raises ValueError: if an unrecognised level key is supplied or if a
        ``'l?'`` dict contains more than one key
    """
    if classes is None:
        return data
    if isinstance(classes, dict):
        classes = [classes]

    allowed_keys = {'l?', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8'}
    all_keys = {k for d in classes for k in d}
    bad_keys = all_keys - allowed_keys
    if bad_keys:
        raise ValueError(f'Unrecognised level key(s): {bad_keys}. Allowed: {sorted(allowed_keys)}')

    lvl_dependent = any('l?' not in d for d in classes)
    lvl_independent = any('l?' in d for d in classes)

    for d in classes:
        if 'l?' in d and len(d) > 1:
            raise ValueError("A dict with 'l?' must contain only that one key.")

    # ------------------------------------------------------------------
    # Build a (targets × classification-levels) DataFrame.
    # ------------------------------------------------------------------
    split_classes = protein_data['Classification'].str.split(';')
    split_classes = equalize_cell_size_in_column(split_classes, 'external', '')
    split_classes = pd.DataFrame(split_classes.tolist())
    multiplicity = len(split_classes.columns)

    for j in range(multiplicity):
        split_classes.iloc[:, j] = split_classes.iloc[:, j].str.split('->')
        split_classes.iloc[:, j] = equalize_cell_size_in_column(
            split_classes.iloc[:, j], 'external', ''
        )
        while len(split_classes.iloc[0, j]) < 8:
            split_classes.iloc[0, j].append('')
        split_classes.iloc[:, j] = equalize_cell_size_in_column(split_classes.iloc[:, j])

    split_classes = pd.concat(
        [
            pd.DataFrame(
                split_classes.iloc[:, j].tolist(),
                columns=[f'l{lvl + 1}_{j + 1}' for lvl in range(8)],
            )
            for j in range(multiplicity)
        ],
        axis=1,
    ).apply(lambda s: s.str.lower())

    # ------------------------------------------------------------------
    # Filter — replace eval() with explicit boolean mask construction.
    # ------------------------------------------------------------------
    mask = pd.Series(False, index=split_classes.index)

    if lvl_dependent:
        for d in (d for d in classes if 'l?' not in d):
            # Each key/value pair must match across all classification slots.
            sub_mask = pd.Series(True, index=split_classes.index)
            for lvl_key, lvl_val in d.items():
                # Any of the multiplicity columns at this level suffices.
                level_cols = [f'{lvl_key}_{j + 1}' for j in range(multiplicity)]
                level_cols = [c for c in level_cols if c in split_classes.columns]
                col_mask = pd.Series(False, index=split_classes.index)
                for col in level_cols:
                    col_mask |= split_classes[col] == lvl_val.lower()
                sub_mask &= col_mask
            mask |= sub_mask

    if lvl_independent:
        for d in (d for d in classes if 'l?' in d):
            pattern = next(iter(d.values()))
            sub_mask = pd.Series(False, index=split_classes.index)
            for col in split_classes.columns:
                if generic_regex:
                    sub_mask |= split_classes[col].str.contains(
                        pattern.lower(), regex=True, na=False
                    )
                else:
                    sub_mask |= split_classes[col].str.lower() == pattern.lower()
            mask |= sub_mask

    indices = split_classes[mask].index.tolist()
    targets = protein_data.loc[indices, 'target_id']
    return data[data['target_id'].isin(targets)].merge(
        protein_data.loc[indices, ['target_id', 'Classification']],
        on='target_id',
    )


@_supports_chunks
def keep_organism(
        data: pd.DataFrame,
        protein_data: pd.DataFrame,
        organism: Optional[Union[str, List[str]]] = 'Homo sapiens (Human)',
        generic_regex: bool = False,
) -> pd.DataFrame:
    """Keep only rows whose target comes from the specified organism(s).

    :param data: bioactivity DataFrame (chunked or not)
    :param protein_data: Papyrus protein-target DataFrame
    :param organism: organism name(s) to retain (case-insensitive).
        ``None`` returns *data* unfiltered.
    :param generic_regex: when True, names are matched as regular expressions
        (partial match); when False, exact match only
    """
    if organism is None:
        return data
    if isinstance(organism, str):
        organism = [organism]

    org_col = protein_data['Organism'].str.lower()
    if generic_regex:
        mask = pd.Series(False, index=protein_data.index)
        for org in organism:
            mask |= org_col.str.contains(org.lower(), regex=True, na=False)
    else:
        mask = org_col.isin(o.lower() for o in organism)

    indices = protein_data[mask].index.tolist()
    targets = protein_data.loc[indices, 'target_id']
    return data[data['target_id'].isin(targets)].merge(
        protein_data.loc[indices, ['target_id', 'Organism']],
        on='target_id',
    )


@_supports_chunks
def keep_match(
        data: pd.DataFrame,
        column: str,
        values: Union[Any, List[Any]],
) -> pd.DataFrame:
    """Keep only rows where *column* is in *values* (equivalent to ``isin``).

    :param data: bioactivity DataFrame (chunked or not)
    :param column: column name to filter on
    :param values: value(s) to retain
    """
    if not isinstance(values, list):
        values = [values]
    return data[data[column].isin(values)]


@_supports_chunks
def keep_not_match(
        data: pd.DataFrame,
        column: str,
        values: Union[Any, List[Any]],
) -> pd.DataFrame:
    """Keep only rows where *column* is **not** in *values* (equivalent to ``~isin``).

    :param data: bioactivity DataFrame (chunked or not)
    :param column: column name to filter on
    :param values: value(s) to exclude
    """
    if not isinstance(values, list):
        values = [values]
    return data[~data[column].isin(values)]


@_supports_chunks
def keep_contains(
        data: pd.DataFrame,
        column: str,
        value: str,
        case: bool = True,
        regex: bool = False,
) -> pd.DataFrame:
    """Keep only rows where *column* contains *value*.

    :param data: bioactivity DataFrame (chunked or not)
    :param column: column name to search in
    :param value: substring or pattern to match
    :param case: whether the match is case-sensitive
    :param regex: whether *value* is a regular expression
    """
    return data[data[column].str.contains(value, case=case, regex=regex)]


@_supports_chunks
def keep_not_contains(
        data: pd.DataFrame,
        column: str,
        value: str,
        case: bool = True,
        regex: bool = False,
) -> pd.DataFrame:
    """Keep only rows where *column* does **not** contain *value*.

    :param data: bioactivity DataFrame (chunked or not)
    :param column: column name to search in
    :param value: substring or pattern to exclude
    :param case: whether the match is case-sensitive
    :param regex: whether *value* is a regular expression
    """
    return data[~data[column].str.contains(value, case=case, regex=regex)]


# ---------------------------------------------------------------------------
# Similarity / substructure helpers
# ---------------------------------------------------------------------------

def _load_fpsubsim2(fpsubsim2_file: str, fingerprint: Optional[Fingerprint]) -> FPSubSim2:
    """Load an :class:`~subsim_search.FPSubSim2` database and validate *fingerprint*.

    :param fpsubsim2_file: path to the ``.h5`` database
    :param fingerprint: fingerprint whose signature must be present in the
        database; pass ``None`` to skip the signature check
    :raises ValueError: if the file does not exist or the fingerprint is absent
    """
    if not os.path.isfile(fpsubsim2_file):
        raise ValueError(f'FPSubSim2 database does not exist: {fpsubsim2_file!r}')
    fpss2 = FPSubSim2()
    fpss2.load(fpsubsim2_file)
    if fingerprint is not None and repr(fingerprint) not in fpss2.available_fingerprints:
        raise ValueError(
            f'FPSubSim2 database does not contain fingerprint {fingerprint.name!r}. '
            f'Available: {list(fpss2.available_fingerprints)}'
        )
    return fpss2


def _collect_similar_molecules(
        fpss2: FPSubSim2,
        molecule_smiles: List[str],
        fingerprint: Fingerprint,
        threshold: float,
        cuda: bool,
) -> pd.DataFrame:
    """Run similarity search for every query SMILES and return merged results.

    :param fpss2: loaded :class:`~subsim_search.FPSubSim2` instance
    :param molecule_smiles: list of SMILES query strings
    :param fingerprint: fingerprint to use for similarity search
    :param threshold: Tanimoto similarity threshold
    :param cuda: use GPU-accelerated search
    :returns: DataFrame with ``InChIKey`` and the similarity score column
    """
    engine = fpss2.get_similarity_lib(fp_signature=repr(fingerprint), cuda=cuda)
    frames = [
        engine.similarity(smi, threshold=threshold)
        for smi in tqdm(molecule_smiles)
    ]
    result = pd.concat(frames, axis=0, ignore_index=True)
    # Return only the identifier and score columns (last two) by name.
    return result[['InChIKey', result.columns[-1]]]


@_supports_chunks
def keep_similar(
        data: pd.DataFrame,
        molecule_smiles: Union[str, List[str]],
        fpsubsim2_file: str,
        fingerprint: Fingerprint = MorganFingerprint(),
        threshold: float = 0.7,
        cuda: bool = False,
) -> pd.DataFrame:
    """Keep only rows associated to molecules similar to the query.

    :param data: bioactivity DataFrame (chunked or not)
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    :param fingerprint: fingerprint to use for similarity search
    :param threshold: Tanimoto similarity threshold
    :param cuda: use GPU-accelerated search
    """
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2 = _load_fpsubsim2(fpsubsim2_file, fingerprint)
    similar_mols = _collect_similar_molecules(fpss2, molecule_smiles, fingerprint, threshold, cuda)
    return data[data['InChIKey'].isin(similar_mols['InChIKey'])].merge(similar_mols, on='InChIKey')


@_supports_chunks
def keep_dissimilar(
        data: pd.DataFrame,
        molecule_smiles: Union[str, List[str]],
        fpsubsim2_file: str,
        fingerprint: Fingerprint = MorganFingerprint(),
        threshold: float = 0.7,
        cuda: bool = False,
) -> pd.DataFrame:
    """Keep only rows associated to molecules **not** similar to the query.

    :param data: bioactivity DataFrame (chunked or not)
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    :param fingerprint: fingerprint to use for similarity search
    :param threshold: Tanimoto similarity threshold
    :param cuda: use GPU-accelerated search
    """
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2 = _load_fpsubsim2(fpsubsim2_file, fingerprint)
    similar_mols = _collect_similar_molecules(fpss2, molecule_smiles, fingerprint, threshold, cuda)
    return data[~data['InChIKey'].isin(similar_mols['InChIKey'])]


def _collect_substructure_molecules(
        fpss2: FPSubSim2,
        molecule_smiles: List[str],
) -> pd.DataFrame:
    """Run substructure search for every query SMILES and return merged results.

    :param fpss2: loaded :class:`~subsim_search.FPSubSim2` instance
    :param molecule_smiles: list of SMILES query strings
    :returns: DataFrame with ``InChIKey`` column
    """
    engine = fpss2.get_substructure_lib()
    frames = [engine.substructure(smi) for smi in tqdm(molecule_smiles)]
    return pd.concat(frames, axis=0, ignore_index=True)


@_supports_chunks
def keep_substructure(
        data: pd.DataFrame,
        molecule_smiles: Union[str, List[str]],
        fpsubsim2_file: str,
) -> pd.DataFrame:
    """Keep only rows associated to substructures of the query molecule(s).

    :param data: bioactivity DataFrame (chunked or not)
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    """
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2 = _load_fpsubsim2(fpsubsim2_file, fingerprint=None)
    substructure_mols = _collect_substructure_molecules(fpss2, molecule_smiles)
    return data[data['InChIKey'].isin(substructure_mols['InChIKey'])]


@_supports_chunks
def keep_not_substructure(
        data: pd.DataFrame,
        molecule_smiles: Union[str, List[str]],
        fpsubsim2_file: str,
) -> pd.DataFrame:
    """Keep only rows associated to molecules that are **not** substructures of the query.

    :param data: bioactivity DataFrame (chunked or not)
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    """
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2 = _load_fpsubsim2(fpsubsim2_file, fingerprint=None)
    substructure_mols = _collect_substructure_molecules(fpss2, molecule_smiles)
    return data[~data['InChIKey'].isin(substructure_mols['InChIKey'])]


# ---------------------------------------------------------------------------
# Chunk consumption
# ---------------------------------------------------------------------------

def consume_chunks(
        generator: Union[PandasTextFileReader, Iterator],
        progress: bool = True,
        total: Optional[int] = None,
) -> pd.DataFrame:
    """Materialise a lazy chain of filters into a single :class:`~pandas.DataFrame`.

    :param generator: iterator produced by one or more chained filter functions
    :param progress: show a tqdm progress bar
    :param total: total number of chunks (for the progress bar)
    :returns: concatenated DataFrame, or an empty DataFrame when the generator
        yields nothing

    .. note::
        Nested generators (filters applied to already-chunked filters) are
        handled recursively.
    """
    frames: List[pd.DataFrame] = []
    iterable = tqdm(generator, total=total) if progress else generator
    for item in iterable:
        if isinstance(item, pd.DataFrame):
            frames.append(item)
        else:
            # Nested generator — recurse without an extra progress bar.
            inner = consume_chunks(item, progress=False)
            if not inner.empty:
                frames.append(inner)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Y-scrambling
# ---------------------------------------------------------------------------

def yscrambling(
        data: Union[pd.DataFrame, PandasTextFileReader, Iterator],
        y_var: Union[str, List[str]] = 'pchembl_value_Mean',
        random_state: int = 1234,
) -> pd.DataFrame:
    """Randomly permute the target variable(s) for y-scrambling experiments.

    :param data: bioactivity DataFrame
    :param y_var: column name(s) to shuffle
    :param random_state: random seed for reproducibility
    :raises ValueError: if *y_var* is not a ``str`` or ``list``
    """
    if not isinstance(y_var, (str, list)):
        raise ValueError('y_var must be a str or a list of str.')
    if not isinstance(y_var, list):
        y_var = [y_var]
    for var in y_var:
        data[var] = shuffle(data[var], random_state=random_state)
    return data
