# -*- coding: utf-8 -*-

"""Filtering functions for the Papyrus dataset."""

import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal, cast

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
#: Mirrors polars' own (non-public) join/concat Literal types.
_JoinHow = Literal['inner', 'left', 'right', 'full', 'semi', 'anti', 'cross', 'outer']
_ConcatHow = Literal[
    'vertical', 'vertical_relaxed', 'diagonal', 'diagonal_relaxed',
    'horizontal', 'horizontal_extend', 'align', 'align_full', 'align_inner',
    'align_left', 'align_right',
]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_data_type(data: object) -> None:
    """Raise ``TypeError`` early (without touching any data) for a bad *data* argument."""
    if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            f'data must be a pl.DataFrame or pl.LazyFrame, '
            f'got {type(data).__name__!r}.',
        )


# ---------------------------------------------------------------------------
# Type-narrowed join/concat helpers
# ---------------------------------------------------------------------------
#
# polars' DataFrame.join()/pl.concat() overloads can't express "same
# concrete type" for a DataFrame | LazyFrame union - narrowing via
# isinstance here does it once, instead of a type: ignore per call site.

def _safe_join(
        left: DataInput,
        right: DataInput,
        *,
        on: str | list[str],
        how: _JoinHow = 'inner',
) -> DataOutput:
    """Join *left* and *right*, which must both be a DataFrame or both a LazyFrame."""
    if isinstance(left, pl.LazyFrame) and isinstance(right, pl.LazyFrame):
        return left.join(right, on=on, how=how)
    if isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame):
        return left.join(right, on=on, how=how)
    raise TypeError(
        f'left and right must both be pl.DataFrame or both pl.LazyFrame, '
        f'got {type(left).__name__} and {type(right).__name__}.',
    )


def _safe_concat(parts: list[DataInput], *, how: _ConcatHow = 'diagonal') -> DataOutput:
    """Concatenate *parts*, which must be all DataFrames or all LazyFrames."""
    # mypy can't narrow a list's element type via a per-element isinstance
    # check (unlike _safe_join's per-variable narrowing) - cast() is still needed.
    if all(isinstance(p, pl.LazyFrame) for p in parts):
        return pl.concat(cast(list[pl.LazyFrame], parts), how=how)
    if all(isinstance(p, pl.DataFrame) for p in parts):
        return pl.concat(cast(list[pl.DataFrame], parts), how=how)
    raise TypeError('parts must be all pl.DataFrame or all pl.LazyFrame.')


# ---------------------------------------------------------------------------
# Papyrus++ column-injection helper
# ---------------------------------------------------------------------------

#: dtypes process_groups() computes for these columns - the raw Papyrus++
#: file ships them pre-existing but as plain String (CSV default), so a row
#: that skips re-aggregation (kept as-is) and a row that goes through
#: process_groups() (genuinely Float64/Int64) would otherwise disagree on
#: dtype when the two are later concatenated.
_PCHEMBL_STAT_DTYPES = {
    'pchembl_value_Mean': pl.Float64,
    'pchembl_value_StdDev': pl.Float64,
    'pchembl_value_SEM': pl.Float64,
    'pchembl_value_N': pl.Int64,
    'pchembl_value_Median': pl.Float64,
    'pchembl_value_MAD': pl.Float64,
}


def _with_papyruspp_columns(data: DataInput) -> tuple[DataInput, list[str]]:
    """Add missing Papyrus++ columns and normalise existing pchembl_value_* dtypes.

    Adds ``Activity_class`` and ``type_other`` null columns when absent, and
    normalises any pre-existing ``pchembl_value_*`` statistic columns to the
    dtypes :func:`process_groups` computes. Returns the (possibly augmented)
    DataFrame/LazyFrame and the list of column names that were added, so
    callers can drop them afterwards.
    """
    schema = data.collect_schema()
    added: list[str] = []
    for col in ('Activity_class', 'type_other'):
        if col not in schema:
            data = data.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
            added.append(col)

    cast_exprs = [
        pl.col(col).cast(dtype, strict=False)
        for col, dtype in _PCHEMBL_STAT_DTYPES.items()
        if col in schema and schema[col] != dtype
    ]
    if cast_exprs:
        data = data.with_columns(cast_exprs)

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
) -> pl.Series | list:
    """Equalise the number of values in each list-containing cell of a column.

    Fully vectorised via Polars' native ``list`` expressions when *col* is
    already a list-typed :class:`~polars.Series` (e.g. the result of
    ``Series.str.split``) or a uniformly scalar-typed one - the common cases.
    A plain Python list mixing scalar and list elements still needs one
    linear pass to tell the two apart (Polars has no dtype for that), but
    that pass is a cheap type check, not the row-by-row padding/truncation
    the previous implementation did in Python.

    :param col: a :class:`~polars.Series` or list whose elements are lists
    :param fill_mode: ``'internal'`` repeats the last value, ``'external'``
        repeats *fill_value*, ``'trim'`` truncates to the shortest list
    :param fill_value: value used when *fill_mode* is ``'external'``
    :raises ValueError: if *fill_mode* is not recognised
    """
    is_series_in = isinstance(col, pl.Series)
    series: pl.Series = col if isinstance(col, pl.Series) else pl.Series(col, strict=False)

    if series.dtype == pl.Object:
        # Genuinely mixed scalar/list Python data: the dtype alone can't
        # distinguish scalars from lists, so one linear pass is unavoidable.
        values = series.to_list()
        mixed_lengths = [len(v) if isinstance(v, list) else 1 for v in values]
        if len(set(mixed_lengths)) == 1:
            return series if is_series_in else values
        series = pl.Series([v if isinstance(v, list) else [v] for v in values])
    elif series.dtype != pl.List:
        # Uniformly scalar-typed: every cell trivially has length 1, so the
        # "already equal" shortcut always holds.
        return series if is_series_in else series.to_list()

    lengths = series.list.len()
    if lengths.len() == 0 or lengths.n_unique() == 1:
        return series if is_series_in else series.to_list()

    # lengths is a non-empty Series of list-lengths (checked above), so
    # .max()/.min() always yield a genuine int - narrower than the broad
    # scalar union polars' stubs declare for a generic Series.
    max_len = cast(int, lengths.max())
    min_len = cast(int, lengths.min())

    if fill_mode == 'trim':
        result = series.list.head(min_len)
    elif fill_mode == 'external':
        pad = pl.Series([[fill_value] * max_len])
        result = series.list.concat(pad).list.head(max_len)
    elif fill_mode == 'internal':
        last = series.list.last()
        pad = pl.DataFrame({'x': last}).select(
            pl.concat_list([pl.col('x')] * max_len).alias('pad'),
        )['pad']
        result = series.list.concat(pad).list.head(max_len)
    else:
        raise ValueError("fill_mode must be one of ['internal', 'external', 'trim']")

    return result if is_series_in else result.to_list()


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


def process_groups(
    data: DataInput,
    additional_columns: list[str] | None = None,
) -> DataOutput:
    """Aggregate duplicate activity records, grouping by ``Activity_ID``.

    Expressed entirely as a single ``group_by().agg()`` call so it stays
    lazy end-to-end when *data* is a :class:`~polars.LazyFrame`.

    :param data: DataFrame or LazyFrame whose rows share ``Activity_ID``
        values to aggregate
    :param additional_columns: extra columns to include with ``listvals`` logic
    :returns: aggregated DataFrame/LazyFrame with ``pchembl_value_Mean``,
        ``_StdDev``, ``_SEM``, ``_N``, ``_Median``, ``_MAD`` columns added
    """
    schema = data.collect_schema()
    listvals_cols = [c for c in (_LISTVALS_COLS + (additional_columns or [])) if c in schema]
    has_pchembl   = 'pchembl_value' in schema

    agg_exprs = [
        pl.when(pl.col(col).n_unique() == 1)
        .then(pl.col(col).cast(pl.Utf8).first())
        .otherwise(pl.col(col).cast(pl.Utf8).str.join(';'))
        .alias(col)
        for col in listvals_cols
    ]

    if has_pchembl:
        pv     = pl.col('pchembl_value')
        # Raw values can carry padding (e.g. "  6.700") - strip before the
        # strict numeric cast, which otherwise rejects it.
        pv_num = pv.str.strip_chars().cast(pl.Float64)
        median = pv_num.median()
        n      = pv_num.drop_nulls().len()
        std    = pv_num.std()
        agg_exprs.extend([
            pv.cast(pl.Utf8).str.join(';').alias('pchembl_value'),
            pv_num.mean().alias('pchembl_value_Mean'),
            std.alias('pchembl_value_StdDev'),
            (std / n.sqrt()).alias('pchembl_value_SEM'),
            n.cast(pl.Int64).alias('pchembl_value_N'),
            median.alias('pchembl_value_Median'),
            (pv_num - median).abs().median().alias('pchembl_value_MAD'),
        ])

    return data.group_by('Activity_ID', maintain_order=True).agg(agg_exprs)


# Keep the single-group entry point for backward compatibility.
def process_group(
    group: DataInput,
    additional_columns: list[str] | None = None,
) -> DataOutput:
    """Aggregate a single Activity-ID group.

    :param group: sub-DataFrame/LazyFrame sharing the same ``Activity_ID``
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

def _validate_split_lengths(split: DataInput, split_cols: list[str], row_max_len: pl.Expr) -> None:
    """Raise ``ValueError`` if repeat-last padding would misattribute a value.

    Safe only when each *split_cols* column's per-row list length is 1
    (shared value) or ``row_max_len`` (full multiplicity); anything in
    between means the columns don't correspond element-for-element. Runs one
    bounded (``limit(1)``) probe, so this stays cheap even on a
    :class:`~polars.LazyFrame`.
    """
    bad_len = pl.any_horizontal([
        (pl.col(c).list.len() != 1) & (pl.col(c).list.len() != row_max_len)
        for c in split_cols
    ])
    probe = (
        split.filter(bad_len)
        .select('Activity_ID', *[pl.col(c).list.len().alias(f'{c}_len') for c in split_cols])
        .limit(1)
    )
    probe_df = probe.collect() if isinstance(probe, pl.LazyFrame) else probe
    if probe_df.height:
        bad = probe_df.row(0, named=True)
        raise ValueError(
            '_unnest_and_filter: found a row whose semicolon-separated columns disagree on '
            'length in a way repeat-last padding cannot safely resolve (each column must be '
            f'either a single shared value or already match the row\'s max multiplicity): {bad}. '
            'Pass validate=False to skip this check if you are certain the input is safe.',
        )


def _unnest_and_filter(
        df: DataInput,
        keep_mask: Callable[[], pl.Expr],
        ordered_columns: list[str],
        aggregate: bool = True,
        additional_columns: list[str] | None = None,
        validate: bool = True,
) -> DataOutput:
    """Split semicolon-joined columns, filter rows, and optionally re-aggregate.

    Stays lazy end-to-end when *df* is a :class:`~polars.LazyFrame`: no row
    count or emptiness is inspected, so the whole pipeline (explode, filter,
    optional aggregation, join) is only ever evaluated once the caller
    collects the final result, except for the bounded validation probe (see
    *validate*).

    :param df: records with semicolon-separated multi-values to process
    :param keep_mask: zero-argument callable returning the filter expression
        (an :class:`~polars.Expr`, not a materialised mask) to apply after exploding
    :param ordered_columns: original column order to restore after merging
    :param aggregate: re-aggregate on ``Activity_ID`` after filtering
    :param additional_columns: forwarded to :func:`process_groups`
    :param validate: run :func:`_validate_split_lengths` first; disable to
        skip that extra probe query if you're certain the input is safe
    """
    schema     = df.collect_schema()
    split_cols = [c for c in _COLS_TO_SPLIT if c in schema]
    excl_cols  = [c for c in schema.names()
                  if c not in split_cols and not c.startswith('pchembl_value_')]

    excluded = df.select(['Activity_ID'] + [c for c in excl_cols if c != 'Activity_ID'])

    # Split each semicolon-delimited column into a list, then pad every
    # column to the row's own max length by repeating its last value
    # (checked safe by _validate_split_lengths() below) - explode() requires
    # matching per-row list lengths across the columns it explodes together.
    # A null cell (e.g. type_other absent) splits to null, not a length-1
    # list - fill_null([None]) gives it a length so it pads/explodes
    # correctly instead of breaking explode's length-agreement requirement.
    split = (
        df.select(['Activity_ID'] + split_cols)
        .with_columns([
            pl.col(c).cast(pl.Utf8).str.split(';').fill_null([None])
            for c in split_cols
        ])
    )
    row_max_len = pl.max_horizontal([pl.col(c).list.len() for c in split_cols])
    if validate:
        _validate_split_lengths(split, split_cols, row_max_len)
    included = (
        split.with_columns([
            pl.col(c).list.concat(pl.col(c).list.last().repeat_by(row_max_len)).list.head(row_max_len)
            for c in split_cols
        ])
        .explode(split_cols, empty_as_null=True)
        # On a genuinely empty frame, the padding expressions above lose
        # their inner dtype (infer List(Null) regardless of upstream casts,
        # a Polars empty-input quirk) - re-cast after exploding, where a
        # cast reliably sticks even with zero rows.
        .with_columns([pl.col(c).cast(pl.Utf8, strict=False) for c in split_cols])
    )

    included = included.filter(keep_mask())

    # included and excluded are both derived from the same df, so they are
    # guaranteed to share the same concrete type - see _safe_join.
    if not aggregate:
        result = _safe_join(included, excluded, on='Activity_ID')
        result_schema = result.collect_schema()
        return result.select([c for c in ordered_columns if c in result_schema])

    aggregated = process_groups(included, additional_columns)
    result     = _safe_join(aggregated, excluded, on='Activity_ID')
    result_schema = result.collect_schema()
    final_cols = [c for c in ordered_columns if c in result_schema]
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
    _validate_data_type(data)
    qualities = ['low', 'medium', 'high']
    if min_quality.lower() not in qualities:
        raise ValueError(f'min_quality must be one of {qualities}, got {min_quality!r}')
    threshold = qualities.index(min_quality.lower())
    return data.filter(pl.col('Quality').str.to_lowercase().is_in(qualities[threshold:]))


def keep_source(
        data: DataInput,
        source: list[str] | str = 'all',
        validate: bool = True,
) -> DataOutput:
    """Keep only rows from the specified data source(s).

    Aggregated statistics (mean, median, SEM …) are recomputed to reflect
    only the retained sources. Fully lazy: nothing is collected, so *data*
    can be a :class:`~polars.LazyFrame` and this stays chainable with other
    filters, evaluating only when the caller finally collects - except for a
    bounded validation probe, see *validate*.

    :param data: bioactivity DataFrame or LazyFrame
    :param source: source label(s) to retain; ``'all'`` or ``'any'`` keeps
        every source
    :param validate: see :func:`_unnest_and_filter`'s *validate* parameter
    """
    _validate_data_type(data)
    if isinstance(source, str):
        source = [source]
    source = [s.lower() for s in source]

    if 'any' in source or 'all' in source:
        return data

    ordered_columns = list(data.collect_schema())
    data, added     = _with_papyruspp_columns(data)
    source_pattern  = '|'.join(map(re.escape, source))
    # Real source values carry version/year suffixes (e.g. 'ChEMBL34',
    # 'Christmann2016'), so matching must be substring-based throughout -
    # not just for detecting whether a multi-source row is worth unnesting.
    matches = pl.col('source').str.to_lowercase().str.contains(source_pattern)
    is_multi = pl.col('source').str.contains(';')

    # Binary-class records with a single matching source — keep as-is.
    preserved_binary = data.filter(
        pl.col('Activity_class').is_not_null() & ~is_multi & matches,
    )
    # Binary-class records with multiple sources — must be unnested first.
    multi_binary = data.filter(
        pl.col('Activity_class').is_not_null() & is_multi & matches,
    )
    # Continuous records.
    cont = data.filter(pl.col('Activity_class').is_null())

    binary_data = _unnest_and_filter(
        multi_binary.drop(added),
        keep_mask=lambda: pl.col('source').str.to_lowercase().str.contains(source_pattern),
        ordered_columns=[c for c in ordered_columns if c not in added],
        aggregate=False,
        validate=validate,
    )

    preserved = cont.filter(~is_multi & matches)
    multi_cont = cont.filter(is_multi & matches)

    filtered = _unnest_and_filter(
        multi_cont.drop(added),
        keep_mask=lambda: pl.col('source').str.to_lowercase().str.contains(source_pattern),
        ordered_columns=[c for c in ordered_columns if c not in added],
        aggregate=True,
        validate=validate,
    )

    # All four parts are derived from the same data, so they are
    # guaranteed to share the same concrete type - see _safe_concat.
    parts = [preserved.drop(added), filtered, preserved_binary.drop(added), binary_data]
    return _safe_concat(parts)


def _activity_type_expr(type_cols: list[str], schema) -> pl.Expr:
    """Expression: exactly one of *type_cols* is ``'1'`` and none is multi-valued."""
    cols = [c for c in type_cols if c in schema]
    if not cols:
        return pl.lit(False)
    has_one  = pl.any_horizontal([pl.col(c).cast(pl.Utf8) == '1' for c in cols])
    no_multi = pl.all_horizontal([~pl.col(c).cast(pl.Utf8).str.contains(';') for c in cols])
    return has_one & no_multi


def _multiple_types_expr(type_cols: list[str], schema) -> pl.Expr:
    """Expression: any of *type_cols* holds a semicolon-separated multi-value."""
    cols = [c for c in type_cols if c in schema]
    if not cols:
        return pl.lit(False)
    return pl.any_horizontal([pl.col(c).cast(pl.Utf8).str.contains(';') for c in cols])


def keep_type(
        data: DataInput,
        activity_types: list[str] | str = 'ic50',
        validate: bool = True,
) -> DataOutput:
    """Keep only rows matching the desired activity type(s).

    Aggregated statistics are recomputed to reflect only the retained types.
    Fully lazy: see :func:`keep_source`.

    :param data: bioactivity DataFrame or LazyFrame
    :param activity_types: type(s) to retain: ``'IC50'``, ``'EC50'``,
        ``'KD'``, ``'Ki'``, ``'other'``, ``'all'``, or ``'any'``
    :param validate: see :func:`_unnest_and_filter`'s *validate* parameter
    :raises ValueError: if any supplied type is not recognised
    """
    _validate_data_type(data)
    canonical = ['IC50', 'EC50', 'KD', 'Ki', 'other']
    lower_map = {t.lower(): t for t in canonical}

    if isinstance(activity_types, str):
        activity_types = [activity_types]
    types_lower = {t.lower() for t in activity_types}

    if 'any' in types_lower or 'all' in types_lower or types_lower >= set(lower_map):
        return data

    unknown = types_lower - set(lower_map)
    if unknown:
        raise ValueError(f'Unrecognised activity type(s): {unknown}. Must be one of {canonical}')

    type_cols       = [f'type_{lower_map[t]}' for t in types_lower]
    ordered_columns = list(data.collect_schema())
    data, added     = _with_papyruspp_columns(data)
    schema          = data.collect_schema()

    is_type  = _activity_type_expr(type_cols, schema)
    is_multi = _multiple_types_expr(type_cols, schema)

    activity_notnull = pl.col('Activity_class').is_not_null()
    activity_isnull  = pl.col('Activity_class').is_null()

    preserved_binary = data.filter(activity_notnull & is_type)
    multi_binary     = data.filter(activity_notnull & is_multi)
    cont             = data.filter(activity_isnull)

    binary_data = _unnest_and_filter(
        multi_binary.drop(added),
        keep_mask=lambda: _activity_type_expr(type_cols, schema),
        ordered_columns=[c for c in ordered_columns if c not in added],
        aggregate=False,
        validate=validate,
    )

    preserved  = cont.filter(is_type)
    multi_cont = cont.filter(is_multi)

    filtered = _unnest_and_filter(
        multi_cont.drop(added),
        keep_mask=lambda: _activity_type_expr(type_cols, schema),
        ordered_columns=[c for c in ordered_columns if c not in added],
        aggregate=True,
        validate=validate,
    )

    # All four parts are derived from the same data, so they are
    # guaranteed to share the same concrete type - see _safe_concat.
    parts = [preserved.drop(added), filtered, preserved_binary.drop(added), binary_data]
    return _safe_concat(parts)


def keep_accession(
        data: DataInput,
        accession: list[str] | str = 'all',
) -> DataOutput:
    """Keep only rows whose target ID matches the given UniProt accession(s).

    :param data: bioactivity DataFrame or LazyFrame
    :param accession: accession code(s) to retain, e.g. ``'P30542'``;
        ``'all'`` or ``'any'`` keeps every accession.
        Mutation suffixes are supported (e.g. ``'P30542_V52A'``).
    """
    if isinstance(accession, str):
        accession = [accession]
    accession_lower = [a.lower() for a in accession]
    if 'all' in accession_lower or 'any' in accession_lower:
        return data
    pattern = '|'.join(re.escape(a) for a in accession_lower)
    return data.filter(pl.col('target_id').str.to_lowercase().str.contains(pattern))


def keep_protein_class(
        data: DataInput,
        protein_data: pl.DataFrame,
        classes: dict | list[dict] | None = None,
        generic_regex: bool = False,
) -> DataOutput:
    """Keep only rows whose target belongs to the desired protein class(es).

    Fully lazy with respect to *data* (never collected): only *protein_data*,
    which is always a small, already-materialised reference table, is
    processed eagerly to build the set of matching target IDs.

    :param data: bioactivity DataFrame or LazyFrame
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
    _validate_data_type(data)
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
    # Passing the split Series directly (not .to_list()) lets
    # equalize_cell_size_in_column take its fully-vectorised List-dtype path.
    classifications = equalize_cell_size_in_column(
        protein_data['Classification'].str.split(';'), 'external', '',
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

    # level_frames always share len(protein_data) rows, so this never pads.
    split_classes = pl.concat(level_frames, how='horizontal_extend')

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

    matched = protein_data.filter(mask).select(['target_id', 'Classification'])
    target_ids = matched['target_id']
    target_df: pl.DataFrame | pl.LazyFrame = matched
    # A LazyFrame can only be joined against another LazyFrame.
    if isinstance(data, pl.LazyFrame):
        target_df = target_df.lazy()
    return _safe_join(data.filter(pl.col('target_id').is_in(target_ids.implode())), target_df, on='target_id')


def keep_organism(
        data: DataInput,
        protein_data: pl.DataFrame,
        organism: str | list[str] | None = 'Homo sapiens (Human)',
        generic_regex: bool = False,
) -> DataOutput:
    """Keep only rows whose target comes from the specified organism(s).

    Fully lazy with respect to *data* (never collected): see
    :func:`keep_protein_class`.

    :param data: bioactivity DataFrame or LazyFrame
    :param protein_data: Papyrus protein-target DataFrame
    :param organism: organism name(s) to retain (case-insensitive).
        ``None`` returns *data* unfiltered.
    :param generic_regex: when True, names are matched as regular expressions
        (partial match); when False, exact match only
    """
    _validate_data_type(data)
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

    matched = protein_data.filter(mask).select(['target_id', 'Organism'])
    matched_target_ids = matched['target_id']
    organism_df: pl.DataFrame | pl.LazyFrame = matched
    # A LazyFrame can only be joined against another LazyFrame.
    if isinstance(data, pl.LazyFrame):
        organism_df = organism_df.lazy()
    return _safe_join(
        data.filter(pl.col('target_id').is_in(matched_target_ids.implode())),
        organism_df,
        on='target_id',
    )


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
            f'Available: {list(fpss2.available_fingerprints)}',
        )
    return fpss2


def _collect_similar_molecules(
        fpss2: FPSubSim2,
        molecule_smiles: list[str],
        fingerprint: Fingerprint,
        threshold: float,
        cuda: bool,
        desc: str = 'Searching similar molecules',
) -> pl.DataFrame:
    """Run similarity search for every query SMILES and return merged results.

    :param fpss2: loaded :class:`~subsim_search.FPSubSim2` instance
    :param molecule_smiles: list of SMILES query strings
    :param fingerprint: fingerprint to use for similarity search
    :param threshold: Tanimoto similarity threshold
    :param cuda: use GPU-accelerated search
    :param desc: progress bar description
    :returns: DataFrame with ``InChIKey`` and the similarity score column
    """
    engine = fpss2.get_similarity_lib(fp_signature=repr(fingerprint), cuda=cuda)
    frames = [
        engine.similarity(smi, threshold=threshold)
        for smi in tqdm(molecule_smiles, desc=desc)
    ]
    result = pl.concat(frames, how='diagonal')
    return result.select(['InChIKey', result.columns[-1]])


def keep_similar(
        data: DataInput,
        molecule_smiles: str | list[str],
        fpsubsim2_file: str | Path,
        fingerprint: Fingerprint | None = None,
        threshold: float = 0.7,
        cuda: bool = False,
) -> DataOutput:
    """Keep only rows associated to molecules similar to the query.

    :param data: bioactivity DataFrame or LazyFrame
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    :param fingerprint: fingerprint to use for similarity search; defaults to ``MorganFingerprint()``
    :param threshold: Tanimoto similarity threshold
    :param cuda: use GPU-accelerated search
    """
    if fingerprint is None:
        fingerprint = MorganFingerprint()
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2        = _load_fpsubsim2(fpsubsim2_file, fingerprint)
    similar_mols = _collect_similar_molecules(
        fpss2, molecule_smiles, fingerprint, threshold, cuda,
        desc='Searching similar molecules',
    )
    score_col    = similar_mols.columns[-1]
    scores: pl.DataFrame | pl.LazyFrame = similar_mols.select(['InChIKey', score_col])
    # A LazyFrame can only be joined against another LazyFrame.
    if isinstance(data, pl.LazyFrame):
        scores = scores.lazy()
    return _safe_join(
        data.filter(pl.col('InChIKey').is_in(similar_mols['InChIKey'].implode())),
        scores,
        on='InChIKey',
    )


def keep_dissimilar(
        data: DataInput,
        molecule_smiles: str | list[str],
        fpsubsim2_file: str | Path,
        fingerprint: Fingerprint | None = None,
        threshold: float = 0.7,
        cuda: bool = False,
) -> DataOutput:
    """Keep only rows associated to molecules **not** similar to the query.

    :param data: bioactivity DataFrame or LazyFrame
    :param molecule_smiles: query SMILES string(s)
    :param fpsubsim2_file: path to the FPSubSim2 ``.h5`` database
    :param fingerprint: fingerprint to use for similarity search; defaults to ``MorganFingerprint()``
    :param threshold: Tanimoto similarity threshold
    :param cuda: use GPU-accelerated search
    """
    if fingerprint is None:
        fingerprint = MorganFingerprint()
    if isinstance(molecule_smiles, str):
        molecule_smiles = [molecule_smiles]
    fpss2        = _load_fpsubsim2(fpsubsim2_file, fingerprint)
    similar_mols = _collect_similar_molecules(
        fpss2, molecule_smiles, fingerprint, threshold, cuda,
        desc='Searching dissimilar molecules',
    )
    return data.filter(~pl.col('InChIKey').is_in(similar_mols['InChIKey'].implode()))


def _collect_substructure_molecules(
        fpss2: FPSubSim2,
        molecule_smiles: list[str],
        desc: str = 'Searching substructure matches',
) -> pl.DataFrame:
    """Run substructure search for every query SMILES and return merged results."""
    engine = fpss2.get_substructure_lib()
    frames = [
        engine.substructure(smi)
        for smi in tqdm(molecule_smiles, desc=desc)
    ]
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
    substructure_mols = _collect_substructure_molecules(
        fpss2, molecule_smiles, desc='Searching substructure matches',
    )
    return data.filter(pl.col('InChIKey').is_in(substructure_mols['InChIKey'].implode()))


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
    substructure_mols = _collect_substructure_molecules(
        fpss2, molecule_smiles, desc='Searching non-substructure matches',
    )
    return data.filter(~pl.col('InChIKey').is_in(substructure_mols['InChIKey'].implode()))


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

def consume_chunks(
        generator: pl.DataFrame | pl.LazyFrame | Iterator,
        progress: bool = True,
        total: int | None = None,
) -> pl.DataFrame:
    """Materialise a lazy frame or a generator of DataFrames into one DataFrame.

    * :class:`~polars.DataFrame` → returned as-is (already materialised).
    * :class:`~polars.LazyFrame` → collected in one call via ``.collect()``.
    * Generator of :class:`~polars.DataFrame` chunks → concatenated.

    :param generator: DataFrame, lazy frame, or iterator produced by one or
        more chained filter functions
    :param progress: show a tqdm progress bar (only for generators)
    :param total: total number of chunks (for the progress bar)
    :returns: concatenated DataFrame, or an empty DataFrame when the generator
        yields nothing
    """
    if isinstance(generator, pl.DataFrame):
        return generator
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
