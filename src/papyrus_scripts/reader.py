# -*- coding: utf-8 -*-

"""Reading functions for the Papyrus dataset."""

import os
from functools import reduce
from pathlib import Path
from collections.abc import Generator

import polars as pl
from prodec import Descriptor, Transform
from tqdm.auto import tqdm

from .utils.IO import (
    PapyrusVersion,
    convert_xz_to_parquet,
    load_data_type_schemas,
    locate_file,
    papyrus_version_module,
    process_data_version,
)
from .utils.mol_reader import MolSupplier


def _prefer_parquet(files: list[Path]) -> Path:
    """Return the ``.parquet`` file among *files* if one is present, else the first match.

    ``download_papyrus`` converts tabular ``.tsv.xz`` files to ``.parquet``
    by default and deletes the ``.xz`` original, but with ``keep_xz=True``
    (or data downloaded before this conversion existed) only the compressed
    original is present - both are matched by the same ``locate_file``
    patterns since they share the ``.tsv`` stem, so the caller must pick.
    """
    for f in files:
        if f.suffix == '.parquet':
            return f
    return files[0]


def _scan_tabular(filepath: Path, **read_kw) -> pl.LazyFrame:
    """Return a lazy scan of a Papyrus tabular file.

    Scans a pre-converted ``.parquet`` file directly (dtypes are embedded,
    *read_kw* is unused). A ``.xz`` original with no ``.parquet`` sibling yet
    (data downloaded before this project converted tabular files, or with
    ``keep_xz=True``) is converted once, via the same memory-bounded chunked
    conversion ``download_papyrus`` uses, and the resulting Parquet file is
    scanned lazily instead - handing Polars the whole decompressed CSV
    content in memory (the only way to give ``pl.scan_csv`` ``.xz`` data,
    since it only decompresses ``.gz`` natively) defeats laziness entirely
    and OOMs on multi-GB Papyrus files. ``.gz``/uncompressed originals are
    scanned directly since Polars streams those without materialising the
    whole file first.
    """
    if filepath.suffix == '.parquet':
        return pl.scan_parquet(filepath)
    if filepath.suffix == '.xz':
        parquet_path = filepath.with_suffix('.parquet')
        if not parquet_path.is_file():
            convert_xz_to_parquet(
                filepath, parquet_path,
                separator=read_kw.get('separator', '\t'),
                schema_overrides=read_kw.get('schema_overrides'),
                null_values=read_kw.get('null_values'),
                progress=True,
            )
        return pl.scan_parquet(parquet_path)
    # Default quoting (quote_char='"') is deliberately left enabled - some
    # Papyrus columns (InChI_AuxInfo, doc_id/citation fields) legitimately
    # hold values with an embedded literal '"' and even a literal newline,
    # properly RFC4180-quoted by the exporter. Disabling quoting doesn't
    # avoid a bug, it causes one: it corrupts those specific records by
    # splitting one logical row into several garbage fragments instead of
    # reconstructing the single multi-line record correctly.
    return pl.scan_csv(filepath, **read_kw)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A single eager DataFrame or a lazy scan (non-greedy).
DataOrChunks = pl.DataFrame | pl.LazyFrame

#: Anything accepted as a ``version`` argument.
VersionArg = str | PapyrusVersion


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


# ---------------------------------------------------------------------------
# Molecular-descriptor helpers
# ---------------------------------------------------------------------------

#: {key: (pattern_tmpl, schema_key_tmpl, dims)} - *dims* lists which of
#: {False (2D), True (3D)} Papyrus actually ships for this descriptor type.
#: mold2/CDDD are 2D-only tools - Papyrus never publishes a 3D variant of
#: either (confirmed against links.json: only '2D_mold2'/'2D_cddd' exist,
#: no '3D_mold2'/'3D_cddd') - so desc_type='all' must skip them for
#: is3d=True instead of unconditionally trying (and failing to find) a file
#: that structurally never exists for this dataset.
_MOL_DESC_REGISTRY = {
    'mold2':       (r'\d+\.\d+_combined_{dim}D_moldescs_mold2\.tsv.*',             'mold2',              (False,)),
    'mordred':     (r'\d+\.\d+_combined_{dim}D_moldescs_mordred{dim_int}D\.tsv.*', 'mordred_{dim_int}D', (False, True)),
    'cddd':        (r'\d+\.\d+_combined_{dim}D_moldescs_CDDDs\.tsv.*',             'CDDD',               (False,)),
    'fingerprint': (r'\d+\.\d+_combined_{dim}D_moldescs_{fp}\.tsv.*',              '{fp}',               (False, True)),
    'moe':         (r'\d+\.\d+_combined_{dim}D_moldescs_MOE\.tsv.*',               None,                 (False, True)),
}

_VALID_DESC_TYPES = frozenset(_MOL_DESC_REGISTRY) | {'all'}


def _resolve_mol_desc_pattern(key: str, is3d: bool):
    pattern_tmpl, schema_key_tmpl, _dims = _MOL_DESC_REGISTRY[key]
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
    picked = _prefer_parquet(files)
    data: pl.LazyFrame = _scan_tabular(picked, **read_kw)
    if ids is not None:
        data = data.filter(pl.col(id_col).is_in(ids))
    return data if lazy else data.collect()


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
    source_mod = papyrus_version_module(pv, root_folder=source_path)
    schema     = load_data_type_schemas(source_mod).get('papyrus', {})

    stereo_tag = 'with' if is3d else 'without'
    pp_tag     = r'\+\+' if plusplus else ''
    pattern    = rf'\d+\.\d+{pp_tag}_combined_set_{stereo_tag}_stereochemistry\.tsv.*'

    filenames = locate_file(source_mod.base, pattern)
    picked    = _prefer_parquet(filenames)
    data      = _scan_tabular(picked, separator='\t', schema_overrides=schema)
    return data if chunksize is not None else data.collect()


def read_protein_set(
    source_path: str | Path | None = None,
    version: VersionArg = 'latest',
) -> pl.DataFrame:
    """Read the protein-target table of the Papyrus dataset.

    :param source_path: root directory for Papyrus data
    :param version: dataset version to read
    """
    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv, root_folder=source_path)

    filenames = locate_file(
        source_mod.base,
        r'\d+\.\d+_combined_set_protein_targets\.tsv.*',
    )
    picked = _prefer_parquet(filenames)
    # null_values=[] keeps empty strings as empty strings (no implicit NA) -
    # only takes effect on the .xz/.gz fallback path; the Parquet file (when
    # present) was already written with the same null_values by download_papyrus.
    return _scan_tabular(picked, separator='\t', null_values=[]).collect()


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
    source_mod = papyrus_version_module(pv, root_folder=source_path)
    schemas    = load_data_type_schemas(source_mod)
    desc_dir   = source_mod.join('descriptors')
    id_col     = 'InChIKey' if is3d else 'connectivity'
    lazy       = chunksize is not None

    if desc_type != 'all':
        return _read_one_mol_descriptor(desc_type, is3d, desc_dir, schemas, lazy, ids, id_col)

    available = [k for k, (_, _, dims) in _MOL_DESC_REGISTRY.items() if is3d in dims]
    all_keys  = [k for k in available if k != 'moe'] + [k for k in available if k == 'moe']
    frames    = [
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
            source_path=papyrus_version_module(pv, root_folder=source_path).base, version=pv,
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
        source_mod = papyrus_version_module(pv, root_folder=source_path)
        schemas    = load_data_type_schemas(source_mod)
        unirep_files = locate_file(
            source_mod.join('descriptors'),
            r'(?:\d+\.\d+_combined_prot_embeddings_unirep\.tsv.*)'
            r'|(?:\d+\.\d+_combined_protdescs_unirep\.tsv.*)',
        )
        return _read_unirep(
            _prefer_parquet(unirep_files),
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
    source_mod = papyrus_version_module(pv, root_folder=source_path)

    stereo_tag = '' if is3d else 'out'
    dim_tag    = 3  if is3d else 2
    pattern    = rf'\d+\.\d+_combined_{dim_tag}D_set_with{stereo_tag}_stereochemistry\.sd.*'

    sd_files = locate_file(source_mod.join('structures'), pattern)
    id_col   = 'InChIKey' if is3d else 'connectivity'

    if chunksize is None:
        return _read_structures_full(sd_files[0], ids, id_col, verbose)
    return _read_structures_chunked(sd_files[0], chunksize, ids, id_col, verbose)


# ---------------------------------------------------------------------------
# Local availability checks
# ---------------------------------------------------------------------------
#
# Cheap, network-free, read-free checks of whether the file(s) a reader
# above would need are already on disk - used by oop.py's _PapyrusSource to
# decide whether a download is needed at all before committing to one
# combined download_papyrus() call for everything a filter chain ends up
# requesting (bioactivity/proteins plus any descriptors/structures), instead
# of one download cycle per file type as each is first touched.

def molecular_descriptors_available(
    desc_type: str,
    is3d: bool = False,
    version: VersionArg = 'latest',
    source_path: str | Path | None = None,
) -> bool:
    """Return whether every file :func:`read_molecular_descriptors` would
    need for *desc_type* is already present on disk, without reading any of
    them.

    :param desc_type: descriptor set; one of ``'mold2'``, ``'mordred'``,
        ``'cddd'``, ``'fingerprint'``, ``'moe'``, ``'all'``
    :param is3d: check for the stereochemistry-aware variant
    :param version: dataset version to check
    :param source_path: root directory for Papyrus data
    :raises ValueError: if *desc_type* is not recognised
    """
    if desc_type not in _VALID_DESC_TYPES:
        raise ValueError(
            f'desc_type must be one of {sorted(_VALID_DESC_TYPES)}, '
            f'got {desc_type!r}'
        )

    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv, root_folder=source_path)
    desc_dir   = source_mod.join('descriptors')

    keys = (
        [k for k, (_, _, dims) in _MOL_DESC_REGISTRY.items() if is3d in dims]
        if desc_type == 'all' else [desc_type]
    )
    for key in keys:
        pattern, _ = _resolve_mol_desc_pattern(key, is3d)
        try:
            locate_file(desc_dir, pattern)
        except (FileNotFoundError, NotADirectoryError):
            return False
    return True


def molecular_structures_available(
    is3d: bool = False,
    version: VersionArg = 'latest',
    source_path: str | Path | None = None,
) -> bool:
    """Return whether the file :func:`read_molecular_structures` would need
    is already present on disk, without reading it.

    :param is3d: check for the stereochemistry-aware (3D) SD file
    :param version: dataset version to check
    :param source_path: root directory for Papyrus data
    """
    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv, root_folder=source_path)

    stereo_tag = '' if is3d else 'out'
    dim_tag    = 3  if is3d else 2
    pattern    = rf'\d+\.\d+_combined_{dim_tag}D_set_with{stereo_tag}_stereochemistry\.sd.*'

    try:
        locate_file(source_mod.join('structures'), pattern)
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False


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
    df = _scan_tabular(Path(filepath), separator='\t', schema_overrides=schema).collect()
    if 'TARGET_NAME' in df.columns:
        df = df.rename({'TARGET_NAME': 'target_id'})
    if ids is not None:
        df = df.filter(pl.col('target_id').is_in(ids))
    return df


def _read_custom_protein_descriptors(
    filepath: str | Path,
    ids: list[str] | None,
) -> pl.DataFrame:
    df = _scan_tabular(Path(filepath), separator='\t').collect()
    if 'TARGET_NAME' in df.columns:
        df = df.rename({'TARGET_NAME': 'target_id'})
    if ids is not None:
        df = df.filter(pl.col('target_id').is_in(ids))
    return df