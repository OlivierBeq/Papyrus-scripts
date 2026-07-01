# -*- coding: utf-8 -*-

"""Filtering functions for the Papyrus dataset."""

import re
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterator

import numpy as np
import polars as pl
from sklearn.utils import shuffle as sk_shuffle
from tqdm.auto import tqdm

from .fingerprint import Fingerprint, MorganFingerprint
from .subsim_search import FPSubSim2

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Any of the two input forms accepted by every public filter.
DataInput = pl.DataFrame | pl.LazyFrame
#: The corresponding output form.
DataOutput = pl.DataFrame | pl.LazyFrame


# ---------------------------------------------------------------------------
# Lazy-dispatch decorator
# ---------------------------------------------------------------------------

def _supports_lazy(fn: Callable) -> Callable:
    """Collect a :class:`~polars.LazyFrame` before passing it to *fn*.

    When the first argument (*data*) is a :class:`~polars.LazyFrame` the
    decorator collects it, calls the original function on the eager
    :class:`~polars.DataFrame`, then re-lazifies the result so that callers
    can keep chaining lazy operations.  For a plain
    :class:`~polars.DataFrame` the function is called directly.

    Simple filters that are expressed entirely as Polars expressions (e.g.
    :func:`keep_quality`) do **not** need this decorator — their
    :meth:`~polars.LazyFrame.filter` calls work on both types natively.
    """

    @wraps(fn)
    def wrapper(data: DataInput, *args, **kwargs) -> DataOutput:
        if isinstance(data, pl.LazyFrame):
            result = fn(data.collect(), *args, **kwargs)
            return result.lazy() if isinstance(result, pl.DataFrame) else result
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f'data must be a pl.DataFrame or pl.LazyFrame, '
                f'got {type(data).__name__!r}.'
            )
        return fn(data, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Papyrus++ column-injection helper
# ---------------------------------------------------------------------------

def _with_papyruspp_columns(data: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Add ``Activity_class`` and ``type_other`` null columns when absent.

    Returns the (possibly augmented) DataFrame and the list of column names
    that were added, so callers can drop them afterwards.
    """
    added: list[str] = []
    for col in ('Activity_class', 'type_other'):
        if col not in data.columns:
            data = data.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
            added.append(col)
    return data, added


# ---------------------------------------------------------------------------
# Cell-size equalisation helpers
# ---------------------------------------------------------------------------

def equalize_cell_size_in_row(
    row: list,
    cols: list[int] | None = None,
    fill_mode: str = 'internal',
    fill_value: object = '',
) -> list:
    """Equalise the number of values in each list-containing cell of a row.

    Operates on a plain Python list (one element per column).

    :param row: list of cell values where list-typed entries are equalised
    :param cols: column indices to equalise; defaults to all columns
    :param fill_mode: ``'internal'`` repeats the last value, ``'external'``
        repeats *fill_value*, ``'trim'`` truncates to the shortest list
    :param fill_value: value used when *fill_mode* is ``'external'``
    :raises ValueError: if *fill_mode* is not recognised
    """
    if cols is None:
        jcols = list(range(len(row)))
    else:
        jcols = list(cols)

    lengths = [len(x) if isinstance(x, list) else 1 for x in row]
    if len(set(lengths)) == 1:
        return row

    vals = [v if isinstance(v, list) else [v] for v in row]
    max_len = max(lengths)
    min_len = min(lengths)

    if fill_mode == 'external':
        vals = [
            e + [fill_value] * (max_len - len(e)) if j in jcols
            else [e[0]] + [fill_value] * (max_len - 1)
            for j, e in enumerate(vals)
        ]
    elif fill_mode == 'internal':
        vals = [
            e + [e[-1]] * (max_len - len(e)) if j in jcols
            else [e[0]] * max_len
            for j, e in enumerate(vals)
        ]
    elif fill_mode == 'trim':
        vals = [e[:min_len] for e in vals]
    else:
        raise ValueError("fill_mode must be one of ['internal', 'external', 'trim']")
    return vals


def equalize_cell_size_in_column(
    col: pl.Series | list,
    fill_mode: str = 'internal',
    fill_value: object = '',
) -> pl.Series:
    """Equalise the number of values in each list-containing cell of a column.

    :param col: a :class:`~polars.Series` or list whose elements are lists
    :param fill_mode: ``'internal'`` repeats the last value, ``'external'``
        repeats *fill_value*, ``'trim'`` truncates to the shortest list
    :param fill_value: value used when *fill_mode* is ``'external'``
    :raises ValueError: if *fill_mode* is not recognised
    """
    values = col.to_list() if isinstance(col, pl.Series) else list(col)
    lengths = [len(x) if isinstance(x, list) else 1 for x in values]

    if len(set(lengths)) == 1:
        return pl.Series(values) if isinstance(col, pl.Series) else values

    vals = [v if isinstance(v, list) else [v] for v in values]
    max_len = max(lengths)
    min_len = min(lengths)

    if fill_mode == 'external':
        result = [e + [fill_value] * (max_len - len(e)) for e in vals]
    elif fill_mode == 'internal':
        result = [e + [e[-1]] * (max_len - len(e)) for e in vals]
    elif fill_mode == 'trim':
        result = [e[:min_len] for e in vals]
    else:
        raise ValueError("fill_mode must be one of ['internal', 'external', 'trim']")

    return pl.Series(result) if isinstance(col, pl.Series) else result


# ---------------------------------------------------------------------------
# Activity-aggregation helpers
# ---------------------------------------------------------------------------

_COLS_TO_SPLIT = [
    'source', 'CID', 'AID',
    'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other',
    'relation', 'pchembl_value',
]

_LISTVALS_COLS = [
    'source', 'CID', 'AID',
    'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'type_other',
    'relation',
]


def _listvals(vals: list) -> str:
    """Return the single value when all equal, else ';'-join all."""
    strs = [str(v) for v in vals]
    return strs[0] if len(set(strs)) == 1 else ';'.join(strs)


def process_groups(
    data: pl.DataFrame,
    additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Aggregate duplicate activity records, grouping by ``Activity_ID``.

    :param data: DataFrame whose rows share ``Activity_ID`` values to aggregate
    :param additional_columns: extra columns to include with ``listvals`` logic
    :returns: aggregated DataFrame with ``pchembl_value_Mean``, ``_StdDev``,
        ``_SEM``, ``_N``, ``_Median``, ``_MAD`` columns added
    """
    listvals_cols = _LISTVALS_COLS + (additional_columns or [])
    has_pchembl   = 'pchembl_value' in data.columns
    cols          = data.columns

    rows: list[dict] = []
    for (act_id,), grp in data.group_by(['Activity_ID'], maintain_order=True):
        row: dict = {'Activity_ID': act_id}

        for col in listvals_cols:
            if col in cols:
                row[col] = _listvals(grp[col].to_list())

        if has_pchembl:
            pv_strs = grp['pchembl_value'].cast(pl.Utf8).to_list()
            row['pchembl_value'] = ';'.join(pv_strs)

            pv_num = grp['pchembl_value'].cast(pl.Float64).drop_nulls()
            n      = len(pv_num)
            mean   = float(pv_num.mean())   if n > 0 else None
            std    = float(pv_num.std())    if n > 1 else None
            med    = float(pv_num.median()) if n > 0 else None
            mad    = float((pv_num - med).abs().median()) if (n > 0 and med is not None) else None
            row['pchembl_value_Mean']   = mean
            row['pchembl_value_StdDev'] = std
            row['pchembl_value_SEM']    = (std / n ** 0.5) if (std is not None and n > 0) else None
            row['pchembl_value_N']      = n
            row['pchembl_value_Median'] = med
            row['pchembl_value_MAD']    = mad

        rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


# Keep the single-group entry point for backward compatibility.
def process_group(
    group: pl.DataFrame,
    additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Aggregate a single Activity-ID group.

    :param group: sub-DataFrame sharing the same ``Activity_ID``
    :param additional_columns: extra columns to include in the aggregation
    """
    return process_groups(group, additional_columns)


# ---------------------------------------------------------------------------
# Row-type helpers  (kept for external callers; logic unchanged)
# ---------------------------------------------------------------------------

def is_activity_type(row: dict, activity_types: list[str]) -> bool:
    """Return True when *row* matches one of the *activity_types* unambiguously.

    :param row: a dict representing a DataFrame row
    :param activity_types: column names (e.g. ``'type_IC50'``) to check
    """
    return (
        any(str(row[t]) == '1' for t in activity_types)
        and all(';' not in str(row[t]) for t in activity_types)
    )


def is_multiple_types(row: dict, activity_types: list[str]) -> bool:
    """Return True when *row* has semicolon-separated values in any *activity_types*.

    :param row: a dict representing a DataFrame row
    :param activity_types: column names to inspect for multi-value entries
    """
    return any(';' in str(row[t]) for t in activity_types)


# ---------------------------------------------------------------------------
# Multi-source / multi-type splitting helper
# ---------------------------------------------------------------------------

def _unnest_and_filter(
        df: pl.DataFrame,
        keep_mask: Callable[[pl.DataFrame], pl.Series],
        ordered_columns: list[str],
        aggregate: bool = True,
        additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Split semicolon-joined columns, filter rows, and optionally re-aggregate.

    :param df: records with semicolon-separated multi-values to process
    :param keep_mask: callable that takes a DataFrame and returns a bool Series
    :param ordered_columns: original column order to restore after merging
    :param aggregate: re-aggregate on ``Activity_ID`` after filtering
    :param additional_columns: forwarded to :func:`process_groups`
    """
    split_cols = [c for c in _COLS_TO_SPLIT if c in df.columns]
    excl_cols  = [c for c in df.columns
                  if c not in split_cols and not c.startswith('pchembl_value_')]

    excluded = df.select(['Activity_ID'] + [c for c in excl_cols if c != 'Activity_ID'])

    # Split each semicolon-delimited column into a list, then explode all at once.
    included = (
        df.select(['Activity_ID'] + split_cols)
        .with_columns([pl.col(c).cast(pl.Utf8).str.split(';') for c in split_cols])
        .explode(split_cols)
    )

    included = included.filter(keep_mask(included))

    if not aggregate or included.is_empty():
        return (
            included
            .join(excluded, on='Activity_ID', how='inner')
            .select([c for c in ordered_columns if c in included.columns or c in excluded.columns])
        )

    aggregated = process_groups(included, additional_columns)
    result     = aggregated.join(excluded, on='Activity_ID', how='inner')
    final_cols = [c for c in ordered_columns if c in result.columns]
    return result.select(final_cols)


# ---------------------------------------------------------------------------
# Public filters
# ---------------------------------------------------------------------------

def keep_quality(
        data: DataInput,
        min_quality: str = 'high',
) -> DataOutput:
    """Keep only rows at or above the minimum required quality level.

    :param data: bioactivity DataFrame or LazyFrame
    :param min_quality: lowest quality to retain: ``'low'``, ``'medium'``,
        or ``'high'``
    :raises ValueError: if *min_quality* is not recognised
    """
    qualities = ['low', 'medium', 'high']
    if min_quality.lower() not in qualities:
        raise ValueError(f'min_quality must be one of {qualities}, got {min_quality!r}')
    threshold = qualities.index(min_quality.lower())
    return data.filter(pl.col('Quality').str.to_lowercase().is_in(qualities[threshold:]))


@_supports_lazy
def keep_source(
        data: pl.DataFrame,
        source: list[str] | str = 'all',
) -> pl.DataFrame:
    """Keep only rows from the specified data source(s).

    Aggregated statistics (mean, median, SEM …) are recomputed to reflect
    only the retained sources.

    :param data: bioactivity DataFrame (or LazyFrame — collected automatically)
    :param source: source label(s) to retain; ``'all'`` or ``'any'`` keeps
        every source
    """
    sources_ = set(chain.from_iterable(s.split(';') for s in data['source'].to_list()))
    sources  = {s.lower() for s in sources_}

    if isinstance(source, str):
        source = [source]
    source = [s.lower() for s in source]

    if 'any' in source or 'all' in source or set(source) >= sources:
        return data

    pattern       = '|'.join(f'^{re.escape(s)}$' for s in source)
    source_adapted = [s for s in sources if re.search(pattern, s)]

    if not source_adapted:
        return data.filter(pl.lit(False))

    ordered_columns = data.columns
    data, added     = _with_papyruspp_columns(data)

    # Binary-class records with a single matching source — keep as-is.
    preserved_binary = data.filter(
        pl.col('Activity_class').is_not_null()
        & pl.col('source').str.to_lowercase().is_in(source_adapted)
    )
    # Binary-class records with multiple sources — must be unnested first.
    multi_binary = data.filter(
        pl.col('Activity_class').is_not_null()
        & pl.col('source').str.contains(';')
        & pl.col('source').str.to_lowercase().str.contains(
            '|'.join(map(re.escape, source_adapted))
        )
    )
    # Continuous records.
    cont = data.filter(pl.col('Activity_class').is_null())

    binary_data = pl.DataFrame()
    if not multi_binary.is_empty():
        binary_data = _unnest_and_filter(
            multi_binary.drop(added),
            keep_mask=lambda df: df['source'].str.to_lowercase().is_in(source_adapted),
            ordered_columns=[c for c in ordered_columns if c not in added],
            aggregate=False,
        )

    preserved = cont.filter(pl.col('source').str.to_lowercase().is_in(source_adapted))
    multi_cont = cont.filter(
        ~pl.col('source').str.to_lowercase().is_in(source_adapted)
        & pl.col('source').str.contains(';')
        & pl.col('source').str.to_lowercase().str.contains(
            '|'.join(map(re.escape, source_adapted))
        )
    )

    filtered = pl.DataFrame()
    if not multi_cont.is_empty():
        filtered = _unnest_and_filter(
            multi_cont.drop(added),
            keep_mask=lambda df: df['source'].str.to_lowercase().is_in(source_adapted),
            ordered_columns=[c for c in ordered_columns if c not in added],
            aggregate=True,
        )

    parts = [df for df in (preserved.drop(added), filtered, preserved_binary.drop(added), binary_data) if not df.is_empty()]
    return pl.concat(parts, how='diagonal') if parts else pl.DataFrame(schema={c: data.schema[c] for c in ordered_columns if c in data.schema})


@_supports_lazy
def keep_type(
        data: pl.DataFrame,
        activity_types: list[str] | str = 'ic50',
) -> pl.DataFrame:
    """Keep only rows matching the desired activity type(s).

    Aggregated statistics are recomputed to reflect only the retained types.

    :param data: bioactivity DataFrame (or LazyFrame — collected automatically)
    :param activity_types: type(s) to retain: ``'IC50'``, ``'EC50'``,
        ``'KD'``, ``'Ki'``, ``'other'``, ``'all'``, or ``'any'``
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

    type_cols       = [f'type_{lower_map[t]}' for t in activity_types]
    ordered_columns = data.columns
    data, added     = _with_papyruspp_columns(data)

    def _is_activity_type(df: pl.DataFrame) -> pl.Series:
        has_one  = pl.Series([False] * len(df))
        no_multi = pl.Series([True]  * len(df))
        for col in type_cols:
            if col in df.columns:
                has_one  = has_one  | (df[col].cast(pl.Utf8) == '1')
                no_multi = no_multi & ~df[col].cast(pl.Utf8).str.contains(';')
        return has_one & no_multi

    def _is_multiple_types(df: pl.DataFrame) -> pl.Series:
        result = pl.Series([False] * len(df))
        for col in type_cols:
            if col in df.columns:
                result = result | df[col].cast(pl.Utf8).str.contains(';')
        return result

    activity_notnull = data['Activity_class'].is_not_null()
    activity_isnull  = data['Activity_class'].is_null()

    preserved_binary = data.filter(activity_notnull & _is_activity_type(data))
    multi_binary     = data.filter(activity_notnull & _is_multiple_types(data))
    cont             = data.filter(activity_isnull)

    binary_data = pl.DataFrame()
    if not multi_binary.is_empty():
        mb = multi_binary.drop(added)
        binary_data = _unnest_and_filter(
            mb,
            keep_mask=_is_activity_type,
            ordered_columns=[c for c in ordered_columns if c not in added],
            aggregate=False,
        )

    preserved  = cont.filter(_is_activity_type(cont))
    multi_cont = cont.filter(_is_multiple_types(cont))

    filtered = pl.DataFrame()
    if not multi_cont.is_empty():
        filtered = _unnest_and_filter(
            multi_cont.drop(added),
            keep_mask=_is_activity_type,
            ordered_columns=[c for c in ordered_columns if c not in added],
            aggregate=True,
        )

    parts = [df for df in (preserved.drop(added), filtered, preserved_binary.drop(added), binary_data) if not df.is_empty()]
    return pl.concat(parts, how='diagonal') if parts else pl.DataFrame(schema={c: data.schema[c] for c in ordered_columns if c in data.schema})


def keep_accession(
        data: DataInput,
        accession: list[str] | str = 'all',
) -> DataOutput:
    """Keep only rows whose target ID matches the given UniProt accession(s).

    :param data: bioactivity DataFrame or LazyFrame
    :param accession: accession code(s) to retain, e.g. ``'P30542'``.
        Mutation suffixes are supported (e.g. ``'P30542_V52A'``).
    """
    if isinstance(accession, str):
        accession = [accession]
    pattern = '|'.join(re.escape(a) for a in accession)
    return data.filter(pl.col('target_id').str.to_lowercase().str.contains(pattern.lower()))


@_supports_lazy
def keep_protein_class(
        data: pl.DataFrame,
        protein_data: pl.DataFrame,
        classes: dict | list[dict] | None = None,
        generic_regex: bool = False,
) -> pl.DataFrame:
    """Keep only rows whose target belongs to the desired protein class(es).

    :param data: bioactivity DataFrame (or LazyFrame — collected automatically)
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
    all_keys     = {k for d in classes for k in d}
    bad_keys     = all_keys - allowed_keys
    if bad_keys:
        raise ValueError(f'Unrecognised level key(s): {bad_keys}. Allowed: {sorted(allowed_keys)}')

    for d in classes:
        if 'l?' in d and len(d) > 1:
            raise ValueError("A dict with 'l?' must contain only that one key.")

    lvl_dependent   = any('l?' not in d for d in classes)
    lvl_independent = any('l?' in d for d in classes)

    # Build a (targets × classification-levels) DataFrame from the protein table.
    classifications = equalize_cell_size_in_column(
        protein_data['Classification'].str.split(';').to_list(), 'external', ''
    )
    if isinstance(classifications, pl.Series):
        classifications = classifications.to_list()

    multiplicity = max(len(c) for c in classifications) if classifications else 0
    # Pad shorter entries
    classifications = [c + [''] * (multiplicity - len(c)) for c in classifications]

    # For each slot, split by '->' into up to 8 levels
    level_frames = []
    for j in range(multiplicity):
        slot = [row[j] for row in classifications]
        levels = [s.split('->') for s in slot]
        levels = [lvl + [''] * (8 - len(lvl)) for lvl in levels]
        cols   = {f'l{lvl + 1}_{j + 1}': [row[lvl].lower() for row in levels] for lvl in range(8)}
        level_frames.append(pl.DataFrame(cols))

    if not level_frames:
        return data.filter(pl.lit(False))

    split_classes = pl.concat(level_frames, how='horizontal')

    # Build the boolean mask over protein rows.
    mask = pl.Series([False] * len(split_classes))

    if lvl_dependent:
        for d in (d for d in classes if 'l?' not in d):
            sub_mask = pl.Series([True] * len(split_classes))
            for lvl_key, lvl_val in d.items():
                level_cols = [f'{lvl_key}_{j + 1}' for j in range(multiplicity)]
                level_cols = [c for c in level_cols if c in split_classes.columns]
                col_mask   = pl.Series([False] * len(split_classes))
                for col in level_cols:
                    col_mask = col_mask | (split_classes[col] == lvl_val.lower())
                sub_mask = sub_mask & col_mask
            mask = mask | sub_mask

    if lvl_independent:
        for d in (d for d in classes if 'l?' in d):
            pattern  = next(iter(d.values()))
            sub_mask = pl.Series([False] * len(split_classes))
            for col in split_classes.columns:
                if generic_regex:
                    sub_mask = sub_mask | split_classes[col].str.contains(pattern.lower())
                else:
                    sub_mask = sub_mask | (split_classes[col] == pattern.lower())
            mask = mask | sub_mask

    matched_indices = [i for i, m in enumerate(mask.to_list()) if m]
    targets         = protein_data['target_id'].gather(matched_indices)
    classification_col = protein_data['Classification'].gather(matched_indices)

    target_df = pl.DataFrame({
        'target_id':      targets,
        'Classification': classification_col,
    })
    return data.filter(pl.col('target_id').is_in(targets)).join(target_df, on='target_id')


@_supports_lazy
def keep_organism(
        data: pl.DataFrame,
        protein_data: pl.DataFrame,
        organism: str | list[str] | None = 'Homo sapiens (Human)',
        generic_regex: bool = False,
) -> pl.DataFrame:
    """Keep only rows whose target comes from the specified organism(s).

    :param data: bioactivity DataFrame (or LazyFrame — collected automatically)
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

    org_col = protein_data['Organism'].str.to_lowercase()
    if generic_regex:
        mask = pl.Series([False] * len(protein_data))
        for org in organism:
            mask = mask | org_col.str.contains(org.lower())
    else:
        mask = org_col.is_in([o.lower() for o in organism])

    matched_indices = [i for i, m in enumerate(mask.to_list()) if m]
    targets         = protein_data['Organism'].gather(matched_indices)

    organism_df = pl.DataFrame({
        'target_id': protein_data['target_id'].gather(matched_indices),
        'Organism':  targets,
    })
    return data.filter(
        pl.col('target_id').is_in(organism_df['target_id'])
    ).join(organism_df, on='target_id')


def keep_match(
        data: DataInput,
        column: str,
        values: Any | list[Any],
) -> DataOutput:
    """Keep only rows where *column* is in *values* (equivalent to ``is_in``).

    :param data: bioactivity DataFrame or LazyFrame
    :param column: column name to filter on
    :param values: value(s) to retain
    """
    if not isinstance(values, list):
        values = [values]
    return data.filter(pl.col(column).is_in(values))


def keep_not_match(
        data: DataInput,
        column: str,
        values: Any | list[Any],
) -> DataOutput:
    """Keep only rows where *column* is **not** in *values*.

    :param data: bioactivity DataFrame or LazyFrame
    :param column: column name to filter on
    :param values: value(s) to exclude
    """
    if not isinstance(values, list):
        values = [values]
    return data.filter(~pl.col(column).is_in(values))


def keep_contains(
        data: DataInput,
        column: str,
        value: str,
        case: bool = True,
        regex: bool = False,
) -> DataOutput:
    """Keep only rows where *column* contains *value*.

    :param data: bioactivity DataFrame or LazyFrame
    :param column: column name to search in
    :param value: substring or pattern to match
    :param case: whether the match is case-sensitive
    :param regex: whether *value* is a regular expression
    """
    expr = pl.col(column)
    if not case:
        expr  = expr.str.to_lowercase()
        value = value.lower()
    return data.filter(expr.str.contains(value, literal=not regex))


def keep_not_contains(
        data: DataInput,
        column: str,
        value: str,
        case: bool = True,
        regex: bool = False,
) -> DataOutput:
    """Keep only rows where *column* does **not** contain *value*.

    :param data: bioactivity DataFrame or LazyFrame
    :param column: column name to search in
    :param value: substring or pattern to exclude
    :param case: whether the match is case-sensitive
    :param regex: whether *value* is a regular expression
    """
    expr = pl.col(column)
    if not case:
        expr  = expr.str.to_lowercase()
        value = value.lower()
    return data.filter(~expr.str.contains(value, literal=not regex))


# ---------------------------------------------------------------------------
# Similarity / substructure helpers
# ---------------------------------------------------------------------------

def _load_fpsubsim2(fpsubsim2_file: str | Path, fingerprint: Fingerprint | None) -> FPSubSim2:
    """Load an :class:`~subsim_search.FPSubSim2` database and validate *fingerprint*.

    :param fpsubsim2_file: path to the ``.h5`` database
    :param fingerprint: fingerprint whose signature must be present in the
        database; pass ``None`` to skip the signature check
    :raises ValueError: if the file does not exist or the fingerprint is absent
    """
    if not Path(fpsubsim2_file).is_file():
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
        molecule_smiles: list[str],
        fingerprint: Fingerprint,
        threshold: float,
        cuda: bool,
) -> pl.DataFrame:
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
    result = pl.concat(frames, how='diagonal')
    return result.select(['InChIKey', result.columns[-1]])


def keep_similar(
        data: DataInput,
        molecule_smiles: str | list[str],
        fpsubsim2_file: str | Path,
        fingerprint: Fingerprint = MorganFingerprint(),
        threshold: float = 0.7,
        cuda: bool = False,
) -> DataOutput:
    """Keep only rows associated to molecules similar to the query.

    :param data: bioactivity DataFrame or LazyFrame
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    :param fingerprint: fingerprint to use for similarity search
    :param threshold: Tanimoto similarity threshold
    :param cuda: use GPU-accelerated search
    """
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2        = _load_fpsubsim2(fpsubsim2_file, fingerprint)
    similar_mols = _collect_similar_molecules(fpss2, molecule_smiles, fingerprint, threshold, cuda)
    score_col    = similar_mols.columns[-1]
    return (
        data.filter(pl.col('InChIKey').is_in(similar_mols['InChIKey']))
        .join(similar_mols.select(['InChIKey', score_col]), on='InChIKey')
    )


def keep_dissimilar(
        data: DataInput,
        molecule_smiles: str | list[str],
        fpsubsim2_file: str | Path,
        fingerprint: Fingerprint = MorganFingerprint(),
        threshold: float = 0.7,
        cuda: bool = False,
) -> DataOutput:
    """Keep only rows associated to molecules **not** similar to the query.

    :param data: bioactivity DataFrame or LazyFrame
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    :param fingerprint: fingerprint to use for similarity search
    :param threshold: Tanimoto similarity threshold
    :param cuda: use GPU-accelerated search
    """
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2        = _load_fpsubsim2(fpsubsim2_file, fingerprint)
    similar_mols = _collect_similar_molecules(fpss2, molecule_smiles, fingerprint, threshold, cuda)
    return data.filter(~pl.col('InChIKey').is_in(similar_mols['InChIKey']))


def _collect_substructure_molecules(
        fpss2: FPSubSim2,
        molecule_smiles: list[str],
) -> pl.DataFrame:
    """Run substructure search for every query SMILES and return merged results."""
    engine = fpss2.get_substructure_lib()
    frames = [engine.substructure(smi) for smi in tqdm(molecule_smiles)]
    return pl.concat(frames, how='diagonal')


def keep_substructure(
        data: DataInput,
        molecule_smiles: str | list[str],
        fpsubsim2_file: str | Path,
) -> DataOutput:
    """Keep only rows associated to substructures of the query molecule(s).

    :param data: bioactivity DataFrame or LazyFrame
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    """
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2             = _load_fpsubsim2(fpsubsim2_file, fingerprint=None)
    substructure_mols = _collect_substructure_molecules(fpss2, molecule_smiles)
    return data.filter(pl.col('InChIKey').is_in(substructure_mols['InChIKey']))


def keep_not_substructure(
        data: DataInput,
        molecule_smiles: str | list[str],
        fpsubsim2_file: str | Path,
) -> DataOutput:
    """Keep only rows associated to molecules that are **not** substructures of the query.

    :param data: bioactivity DataFrame or LazyFrame
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    """
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2             = _load_fpsubsim2(fpsubsim2_file, fingerprint=None)
    substructure_mols = _collect_substructure_molecules(fpss2, molecule_smiles)
    return data.filter(~pl.col('InChIKey').is_in(substructure_mols['InChIKey']))


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

def consume_chunks(
        generator: pl.LazyFrame | Iterator,
        progress: bool = True,
        total: int | None = None,
) -> pl.DataFrame:
    """Materialise a lazy frame or a generator of DataFrames into one DataFrame.

    * :class:`~polars.LazyFrame` → collected in one call via ``.collect()``.
    * Generator of :class:`~polars.DataFrame` chunks → concatenated.

    :param generator: lazy frame or iterator produced by one or more chained
        filter functions
    :param progress: show a tqdm progress bar (only for generators)
    :param total: total number of chunks (for the progress bar)
    :returns: concatenated DataFrame, or an empty DataFrame when the generator
        yields nothing
    """
    if isinstance(generator, pl.LazyFrame):
        return generator.collect()

    frames: list[pl.DataFrame] = []
    iterable = tqdm(generator, total=total) if progress else generator
    for item in iterable:
        if isinstance(item, pl.DataFrame):
            frames.append(item)
        elif isinstance(item, pl.LazyFrame):
            frames.append(item.collect())
        else:
            inner = consume_chunks(item, progress=False)
            if not inner.is_empty():
                frames.append(inner)
    return pl.concat(frames, how='diagonal') if frames else pl.DataFrame()


# ---------------------------------------------------------------------------
# Y-scrambling
# ---------------------------------------------------------------------------

def yscrambling(
        data: DataInput,
        y_var: str | list[str] = 'pchembl_value_Mean',
        random_state: int = 1234,
) -> pl.DataFrame:
    """Randomly permute the target variable(s) for y-scrambling experiments.

    :param data: bioactivity DataFrame or LazyFrame (collected if lazy)
    :param y_var: column name(s) to shuffle
    :param random_state: random seed for reproducibility
    :raises ValueError: if *y_var* is not a ``str`` or ``list``
    """
    if not isinstance(y_var, (str, list)):
        raise ValueError('y_var must be a str or a list of str.')
    if not isinstance(y_var, list):
        y_var = [y_var]
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    for var in y_var:
        shuffled = pl.Series(sk_shuffle(data[var].to_numpy(), random_state=random_state))
        data = data.with_columns(shuffled.alias(var))
    return data