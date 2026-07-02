# -*- coding: utf-8 -*-

"""Reading functions for the Papyrus dataset."""

import json
import os
from functools import reduce
from pathlib import Path
from collections.abc import Generator

import numpy as np
import polars as pl
import pystow
from prodec import Descriptor, Transform
from tqdm.auto import tqdm

from .utils.IO import (
    PapyrusVersion,
    TypeDecoder,
    locate_file,
    papyrus_version_module,
    process_data_version,
)
from .utils.mol_reader import MolSupplier


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A single eager DataFrame or a lazy scan (non-greedy).
DataOrChunks = pl.DataFrame | pl.LazyFrame

#: Anything accepted as a ``version`` argument.
VersionArg = str | PapyrusVersion


# ---------------------------------------------------------------------------
# Dtype-conversion helpers
# ---------------------------------------------------------------------------

_BUILTIN_TO_POLARS: dict = {
    str:   pl.Utf8,
    float: pl.Float64,
    int:   pl.Int64,
    bool:  pl.Boolean,
}

_NUMPY_TO_POLARS: dict = {
    np.float32: pl.Float32,
    np.float64: pl.Float64,
    np.int8:    pl.Int8,
    np.int16:   pl.Int16,
    np.int32:   pl.Int32,
    np.int64:   pl.Int64,
    np.uint8:   pl.UInt8,
    np.uint16:  pl.UInt16,
    np.uint32:  pl.UInt32,
    np.uint64:  pl.UInt64,
    np.bool_:   pl.Boolean,
    np.str_:    pl.Utf8,
    np.object_: pl.Utf8,
}


def _to_polars_dtype(t) -> pl.DataType:
    """Convert a Python builtin or NumPy type to the nearest Polars dtype."""
    if t in _BUILTIN_TO_POLARS:
        return _BUILTIN_TO_POLARS[t]
    for np_type, pl_type in _NUMPY_TO_POLARS.items():
        if t is np_type or (isinstance(t, type) and issubclass(t, np_type)):
            return pl_type
    return pl.Utf8


def _to_polars_schema(dtypes: dict) -> dict:
    """Convert a ``{col: python_type}`` map to a ``{col: polars_dtype}`` schema."""
    return {col: _to_polars_dtype(t) for col, t in dtypes.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_pystow_home(source_path: str | Path | None) -> None:
    """Point pystow at *source_path* when it is not None."""
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = str(Path(source_path).resolve())


def _resolve_version(
    version: VersionArg,
    source_path: str | Path | None,
) -> PapyrusVersion:
    """Set pystow home and validate *version* against local data."""
    _set_pystow_home(source_path)
    return process_data_version(version=version, root_folder=source_path)


def _load_schemas(source_module: pystow.Module) -> dict:
    """Read ``data_types.json`` and return ``{section: {col: polars_dtype}}``."""
    dtype_file = source_module.join(name='data_types.json')
    with open(dtype_file) as fh:
        raw = json.load(fh, cls=TypeDecoder)
    return {
        key: _to_polars_schema(val) if isinstance(val, dict) else val
        for key, val in raw.items()
    }


# ---------------------------------------------------------------------------
# Molecular-descriptor helpers
# ---------------------------------------------------------------------------

_MOL_DESC_REGISTRY = {
    'mold2':       (r'\d+\.\d+_combined_{dim}D_moldescs_mold2\.tsv.*',             'mold2'),
    'mordred':     (r'\d+\.\d+_combined_{dim}D_moldescs_mordred{dim_int}D\.tsv.*', 'mordred_{dim_int}D'),
    'cddd':        (r'\d+\.\d+_combined_{dim}D_moldescs_CDDDs\.tsv.*',             'CDDD'),
    'fingerprint': (r'\d+\.\d+_combined_{dim}D_moldescs_{fp}\.tsv.*',              '{fp}'),
    'moe':         (r'\d+\.\d+_combined_{dim}D_moldescs_MOE\.tsv.*',               None),
}

_VALID_DESC_TYPES = frozenset(_MOL_DESC_REGISTRY) | {'all'}


def _resolve_mol_desc_pattern(key: str, is3d: bool):
    pattern_tmpl, schema_key_tmpl = _MOL_DESC_REGISTRY[key]
    dim     = '3' if is3d else '2'
    dim_int = 3   if is3d else 2
    fp      = 'E3FP' if is3d else 'ECFP6'
    pattern    = pattern_tmpl.format(dim=dim, dim_int=dim_int, fp=fp)
    schema_key = (
        schema_key_tmpl.format(dim=dim, dim_int=dim_int, fp=fp)
        if schema_key_tmpl is not None else None
    )
    return pattern, schema_key


def _read_one_mol_descriptor(
    key: str,
    is3d: bool,
    desc_dir: str | Path,
    schemas: dict,
    lazy: bool,
    ids: list[str] | None,
    id_col: str,
) -> DataOrChunks:
    pattern, schema_key = _resolve_mol_desc_pattern(key, is3d)
    files  = locate_file(desc_dir, pattern)
    schema = schemas.get(schema_key) if schema_key is not None else None
    read_kw: dict = dict(separator='\t')
    if schema:
        read_kw['schema_overrides'] = schema
    data: DataOrChunks = (
        pl.scan_csv(files[0], **read_kw) if lazy
        else pl.read_csv(files[0], **read_kw)
    )
    if ids is not None:
        data = data.filter(pl.col(id_col).is_in(ids))
    return data


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def read_papyrus(
    is3d: bool = False,
    version: VersionArg = 'latest',
    plusplus: bool = True,
    chunksize: int | None = None,
    source_path: str | Path | None = None,
) -> DataOrChunks:
    """Read the Papyrus bioactivity dataset.

    :param is3d: load the stereochemistry-aware (3D) variant (default: False)
    :param version: dataset version to read
    :param plusplus: load the high-quality Papyrus++ subset (default: True)
    :param chunksize: when not ``None``, return a lazy :class:`~polars.LazyFrame`
        instead of loading everything into memory.  The numeric value is no
        longer used as a row count — any non-``None`` value enables lazy mode.
    :param source_path: root directory for Papyrus data
    :raises ValueError: if the 3D Papyrus++ combination is requested
    """
    if is3d and plusplus:
        raise ValueError('Papyrus++ is only available without stereochemistry.')

    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv)
    schema     = _load_schemas(source_mod).get('papyrus', {})

    stereo_tag = 'with' if is3d else 'without'
    pp_tag     = r'\+\+' if plusplus else ''
    pattern    = rf'\d+\.\d+{pp_tag}_combined_set_{stereo_tag}_stereochemistry\.tsv.*'

    filenames = locate_file(source_mod.base, pattern)
    read_kw   = dict(separator='\t', schema_overrides=schema)
    if chunksize is None:
        return pl.read_csv(filenames[0], **read_kw)
    return pl.scan_csv(filenames[0], **read_kw)


def read_protein_set(
    source_path: str | Path | None = None,
    version: VersionArg = 'latest',
) -> pl.DataFrame:
    """Read the protein-target table of the Papyrus dataset.

    :param source_path: root directory for Papyrus data
    :param version: dataset version to read
    """
    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv)

    filenames = locate_file(
        source_mod.base,
        r'\d+\.\d+_combined_set_protein_targets\.tsv.*',
    )
    # null_values=[] keeps empty strings as empty strings (no implicit NA).
    return pl.read_csv(filenames[0], separator='\t', null_values=[])


def read_molecular_descriptors(
    desc_type: str = 'mold2',
    is3d: bool = False,
    version: VersionArg = 'latest',
    chunksize: int | None = None,
    source_path: str | Path | None = None,
    ids: list[str] | None = None,
    verbose: bool = True,
) -> DataOrChunks:
    """Read pre-computed molecular descriptors.

    :param desc_type: descriptor set; one of ``'mold2'``, ``'mordred'``,
        ``'cddd'``, ``'fingerprint'``, ``'moe'``, ``'all'``
    :param is3d: load descriptors for the stereochemistry-aware variant
    :param version: dataset version to read
    :param chunksize: when not ``None``, return a lazy :class:`~polars.LazyFrame`.
        The numeric value is no longer used — any non-``None`` value enables
        lazy mode.
    :param source_path: root directory for Papyrus data
    :param ids: molecule identifiers to retain; ``None`` keeps all
    :param verbose: unused; kept for API compatibility
    :raises ValueError: if *desc_type* is not recognised
    """
    if desc_type not in _VALID_DESC_TYPES:
        raise ValueError(
            f'desc_type must be one of {sorted(_VALID_DESC_TYPES)}, '
            f'got {desc_type!r}'
        )

    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv)
    schemas    = _load_schemas(source_mod)
    desc_dir   = source_mod.join('descriptors')
    id_col     = 'InChIKey' if is3d else 'connectivity'
    lazy       = chunksize is not None

    if desc_type != 'all':
        return _read_one_mol_descriptor(desc_type, is3d, desc_dir, schemas, lazy, ids, id_col)

    all_keys = [k for k in _MOL_DESC_REGISTRY if k != 'moe'] + ['moe']
    frames   = [
        _read_one_mol_descriptor(k, is3d, desc_dir, schemas, lazy, ids, id_col)
        for k in all_keys
    ]
    # Join all descriptor frames on the common identifier column.
    return reduce(lambda a, b: a.join(b, on=id_col, how='inner'), frames)


def read_protein_descriptors(
    desc_type: str | Descriptor | Transform = 'unirep',
    version: VersionArg = 'latest',
    chunksize: int | None = None,
    source_path: str | Path | None = None,
    ids: list[str] | None = None,
    verbose: bool = True,
    **kwargs,
) -> pl.DataFrame:
    """Read protein descriptors.

    :param desc_type: ``'unirep'``, ``'custom'``, or a ProDEC
        :class:`~prodec.Descriptor` / :class:`~prodec.Transform`
    :param version: dataset version to read (ignored for ``'custom'``)
    :param chunksize: currently has no effect; the returned DataFrame is
        always fully materialised. Kept for API stability.
    :param source_path: for ``'unirep'``/ProDEC: root directory for Papyrus
        data.  For ``'custom'``: path to a TSV file.
    :param ids: target identifiers to retain; ``None`` keeps all
    :param verbose: unused; kept for API compatibility
    :param kwargs: extra keyword arguments forwarded to ProDEC ``pandas_get``
    """
    if desc_type == 'custom':
        if not Path(source_path).is_file():
            raise ValueError(
                'source_path must point to an existing file when desc_type="custom"'
            )
        return _read_custom_protein_descriptors(source_path, ids)

    if isinstance(desc_type, (Descriptor, Transform)):
        pv           = _resolve_version(version, source_path)
        protein_data = read_protein_set(
            source_path=papyrus_version_module(pv).base, version=pv,
        )
        protein_data = protein_data.rename({'TARGET_NAME': 'target_id'})
        if ids is not None:
            protein_data = protein_data.filter(pl.col('target_id').is_in(ids))
        protein_data = protein_data.filter(
            pl.col('Sequence').map_elements(
                desc_type.Descriptor.is_sequence_valid, return_dtype=pl.Boolean,
            )
        )
        # ProDEC returns a pandas DataFrame; convert to polars.
        import pandas as _pd
        descriptors: _pd.DataFrame = desc_type.pandas_get(
            protein_data['Sequence'].to_list(),
            protein_data['target_id'].to_list(),
            **kwargs,
        )
        return pl.from_pandas(descriptors.rename(columns={'ID': 'target_id'}))

    if desc_type == 'unirep':
        pv         = _resolve_version(version, source_path)
        source_mod = papyrus_version_module(pv)
        schemas    = _load_schemas(source_mod)
        unirep_files = locate_file(
            source_mod.join('descriptors'),
            r'(?:\d+\.\d+_combined_prot_embeddings_unirep\.tsv.*)'
            r'|(?:\d+\.\d+_combined_protdescs_unirep\.tsv.*)',
        )
        return _read_unirep(
            unirep_files[0],
            schema=schemas.get('unirep', {}),
            ids=ids,
        )

    raise ValueError(
        f'desc_type must be "unirep", "custom", or a ProDEC Descriptor/Transform, '
        f'got {desc_type!r}'
    )


def read_molecular_structures(
    is3d: bool = False,
    version: VersionArg = 'latest',
    chunksize: int | None = None,
    source_path: str | Path | None = None,
    ids: list[str] | None = None,
    verbose: bool = True,
) -> pl.DataFrame | Generator[pl.DataFrame]:
    """Read molecular structures from the Papyrus SD files.

    Returns a :class:`~polars.DataFrame` (``chunksize=None``) or a generator
    of DataFrames (``chunksize`` set).  The ``'mol'`` column holds RDKit
    :class:`~rdkit.Chem.rdchem.Mol` objects stored as a Polars ``Object``
    series.

    :param is3d: load the stereochemistry-aware (3D) SD file
    :param version: dataset version to read
    :param chunksize: molecules per chunk; ``None`` loads all at once
    :param source_path: root directory for Papyrus data
    :param ids: molecule identifiers to retain; ``None`` keeps all
    :param verbose: show a progress bar
    """
    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv)

    stereo_tag = '' if is3d else 'out'
    dim_tag    = 3  if is3d else 2
    pattern    = rf'\d+\.\d+_combined_{dim_tag}D_set_with{stereo_tag}_stereochemistry\.sd.*'

    sd_files = locate_file(source_mod.join('structures'), pattern)
    id_col   = 'InChIKey' if is3d else 'connectivity'

    if chunksize is None:
        return _read_structures_full(sd_files[0], ids, id_col, verbose)
    return _read_structures_chunked(sd_files[0], chunksize, ids, id_col, verbose)


# ---------------------------------------------------------------------------
# Internal structure readers
# ---------------------------------------------------------------------------

def _read_structures_full(
    sd_file: str | Path,
    ids: list[str] | None,
    id_col: str,
    verbose: bool,
) -> pl.DataFrame:
    rows: list = []
    with MolSupplier(sd_file, show_progress=verbose) as supplier:
        for _, mol in supplier:
            props = mol.GetPropsAsDict()
            if ids is not None and props[id_col] not in ids:
                continue
            props['mol'] = mol
            rows.append(props)
    if not rows:
        return pl.DataFrame()
    return pl.from_dicts(rows, schema_overrides={'mol': pl.Object})


def _read_structures_chunked(
    sd_file: str | Path,
    chunksize: int,
    ids: list[str] | None,
    id_col: str,
    verbose: bool,
) -> Generator[pl.DataFrame]:
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError('chunksize must be a positive integer.')

    pbar = tqdm(desc='Loading molecular structures') if verbose else None
    rows: list = []
    try:
        with MolSupplier(sd_file) as supplier:
            for _, mol in supplier:
                props = mol.GetPropsAsDict()
                if ids is not None and props[id_col] not in ids:
                    continue
                props['mol'] = mol
                rows.append(props)
                if len(rows) == chunksize:
                    if pbar is not None:
                        pbar.update()
                    yield pl.from_dicts(rows, schema_overrides={'mol': pl.Object})
                    rows = []
            if pbar is not None:
                pbar.update()
            if rows:
                yield pl.from_dicts(rows, schema_overrides={'mol': pl.Object})
    finally:
        if pbar is not None:
            pbar.close()


# ---------------------------------------------------------------------------
# Internal protein-descriptor readers
# ---------------------------------------------------------------------------

def _read_unirep(
    filepath: str | Path,
    schema: dict,
    ids: list[str] | None,
) -> pl.DataFrame:
    df = pl.read_csv(filepath, separator='\t', schema_overrides=schema)
    if 'TARGET_NAME' in df.columns:
        df = df.rename({'TARGET_NAME': 'target_id'})
    if ids is not None:
        df = df.filter(pl.col('target_id').is_in(ids))
    return df


def _read_custom_protein_descriptors(
    filepath: str | Path,
    ids: list[str] | None,
) -> pl.DataFrame:
    df = pl.read_csv(filepath, separator='\t')
    if 'TARGET_NAME' in df.columns:
        df = df.rename({'TARGET_NAME': 'target_id'})
    if ids is not None:
        df = df.filter(pl.col('target_id').is_in(ids))
    return df