# -*- coding: utf-8 -*-

"""Reading functions for the Papyrus dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Union

import pandas as pd
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

#: A single DataFrame or a lazy generator of DataFrame chunks.
DataOrChunks = Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]

#: Anything accepted as a ``version`` argument.
VersionArg = Union[str, PapyrusVersion]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_pystow_home(source_path: Optional[Union[str, Path]]) -> None:
    """Point pystow at *source_path* when it is not None."""
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(
            source_path if isinstance(source_path, str) else str(source_path)
        )


def _resolve_version(
    version: VersionArg,
    source_path: Optional[Union[str, Path]],
) -> PapyrusVersion:
    """Set pystow home and validate *version* against local data in one step.

    :param version: raw version argument from the caller
    :param source_path: optional root directory for Papyrus data
    :returns: validated :class:`~utils.IO.PapyrusVersion` instance
    """
    _set_pystow_home(source_path)
    # process_data_version is now the single, canonical resolver; it lives in
    # IO.py and is re-exported from there.  reader.py no longer needs its own
    # copy of the logic.
    return process_data_version(version=version, root_folder=source_path)


def _load_dtypes(source_module: pystow.Module) -> dict:
    """Read and return the ``data_types.json`` type-map for *source_module*."""
    dtype_file = source_module.join(name='data_types.json').as_posix()
    with open(dtype_file, 'r') as fh:
        return json.load(fh, cls=TypeDecoder)


def _maybe_tqdm(
    iterable,
    verbose: bool,
    desc: str,
    total: Optional[int] = None,
):
    """Wrap *iterable* in a tqdm progress bar when *verbose* is True.

    Unlike ``partial(tqdm, ...)``, this helper is only applied to iterators
    (chunked readers), never to plain DataFrames.
    """
    return tqdm(iterable, desc=desc, total=total) if verbose else iterable


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
    pattern_tmpl, dtype_key_tmpl = _MOL_DESC_REGISTRY[key]
    dim     = '3' if is3d else '2'
    dim_int = 3   if is3d else 2
    fp      = 'E3FP' if is3d else 'ECFP6'
    pattern   = pattern_tmpl.format(dim=dim, dim_int=dim_int, fp=fp)
    dtype_key = (
        dtype_key_tmpl.format(dim=dim, dim_int=dim_int, fp=fp)
        if dtype_key_tmpl is not None else None
    )
    return pattern, dtype_key


def _read_one_mol_descriptor(
    key: str,
    is3d: bool,
    desc_dir: str,
    dtypes: dict,
    chunksize: Optional[int],
    ids: Optional[List[str]],
    id_col: str,
) -> DataOrChunks:
    pattern, dtype_key = _resolve_mol_desc_pattern(key, is3d)
    files = locate_file(desc_dir, pattern)
    dtype = dtypes.get(dtype_key) if dtype_key is not None else None
    read_kw = dict(sep='\t', low_memory=True, chunksize=chunksize)
    if dtype is not None:
        read_kw['dtype'] = dtype
    raw = pd.read_csv(files[0], **read_kw)
    return _filter_descriptors(raw, ids, id_col, chunksize is not None)


def _filter_descriptors(
    data: Union[pd.DataFrame, 'pd.io.parsers.TextFileReader'],
    ids: Optional[List[str]],
    id_col: str,
    chunked: bool,
) -> DataOrChunks:
    if not chunked:
        return data if ids is None else data[data[id_col].isin(ids)]

    def _gen():
        for chunk in data:
            yield chunk if ids is None else chunk[chunk[id_col].isin(ids)]

    return _gen()


def _join_descriptor_chunks(
    *iters: Iterator[pd.DataFrame],
    on: str,
) -> Generator[pd.DataFrame, None, None]:
    """Zip-join multiple descriptor chunk iterators on a common key column."""
    for chunks in zip(*iters):
        indexed = [chunk.set_index(on) for chunk in chunks]
        merged  = pd.concat(indexed, axis=1)
        merged.reset_index(inplace=True)
        yield merged


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def read_papyrus(
    is3d: bool = False,
    version: VersionArg = 'latest',
    plusplus: bool = True,
    chunksize: Optional[int] = None,
    source_path: Optional[str] = None,
) -> DataOrChunks:
    """Read the Papyrus bioactivity dataset.

    :param is3d: load the stereochemistry-aware (3D) variant (default: False)
    :param version: dataset version to read
    :param plusplus: load the high-quality Papyrus++ subset (default: True)
    :param chunksize: rows per chunk; ``None`` loads everything at once
    :param source_path: root directory for Papyrus data
    :raises ValueError: if the 3D Papyrus++ combination is requested
    """
    if is3d and plusplus:
        raise ValueError('Papyrus++ is only available without stereochemistry.')

    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv)
    dtypes     = _load_dtypes(source_mod)['papyrus']

    stereo_tag = 'with' if is3d else 'without'
    pp_tag     = r'\+\+' if plusplus else ''
    pattern    = rf'\d+\.\d+{pp_tag}_combined_set_{stereo_tag}_stereochemistry\.tsv.*'

    filenames = locate_file(source_mod.base.as_posix(), pattern)
    return pd.read_csv(filenames[0], sep='\t', chunksize=chunksize, dtype=dtypes, low_memory=True)


def read_protein_set(
    source_path: Optional[str] = None,
    version: VersionArg = 'latest',
) -> pd.DataFrame:
    """Read the protein-target table of the Papyrus dataset.

    :param source_path: root directory for Papyrus data
    :param version: dataset version to read
    """
    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv)

    filenames = locate_file(
        source_mod.base.as_posix(),
        r'\d+\.\d+_combined_set_protein_targets\.tsv.*',
    )
    return pd.read_csv(filenames[0], sep='\t', keep_default_na=False)


def read_molecular_descriptors(
    desc_type: str = 'mold2',
    is3d: bool = False,
    version: VersionArg = 'latest',
    chunksize: Optional[int] = None,
    source_path: Optional[str] = None,
    ids: Optional[List[str]] = None,
    verbose: bool = True,
) -> DataOrChunks:
    """Read pre-computed molecular descriptors.

    :param desc_type: descriptor set; one of ``'mold2'``, ``'mordred'``,
        ``'cddd'``, ``'fingerprint'``, ``'moe'``, ``'all'``
    :param is3d: load descriptors for the stereochemistry-aware variant
    :param version: dataset version to read
    :param chunksize: rows per chunk; ``None`` loads everything at once
    :param source_path: root directory for Papyrus data
    :param ids: molecule identifiers to retain; ``None`` keeps all
    :param verbose: show a progress bar when reading chunks
    :raises ValueError: if *desc_type* is not recognised
    """
    if desc_type not in _VALID_DESC_TYPES:
        raise ValueError(
            f'desc_type must be one of {sorted(_VALID_DESC_TYPES)}, '
            f'got {desc_type!r}'
        )

    pv         = _resolve_version(version, source_path)
    source_mod = papyrus_version_module(pv)
    dtypes     = _load_dtypes(source_mod)
    desc_dir   = source_mod.join('descriptors').as_posix()
    id_col     = 'InChIKey' if is3d else 'connectivity'
    chunked    = chunksize is not None

    if desc_type != 'all':
        data = _read_one_mol_descriptor(desc_type, is3d, desc_dir, dtypes, chunksize, ids, id_col)
        if chunked and verbose:
            data = _maybe_tqdm(data, verbose=True, desc='Loading molecular descriptors')
        return data

    all_keys = [k for k in _MOL_DESC_REGISTRY if k != 'moe'] + ['moe']

    if not chunked:
        frames = [
            _read_one_mol_descriptor(k, is3d, desc_dir, dtypes, None, ids, id_col).set_index(id_col)
            for k in all_keys
        ]
        merged = pd.concat(frames, axis=1)
        merged.reset_index(inplace=True)
        return merged

    iters  = [
        _read_one_mol_descriptor(k, is3d, desc_dir, dtypes, chunksize, ids, id_col)
        for k in all_keys
    ]
    joined = _join_descriptor_chunks(*iters, on=id_col)
    if verbose:
        joined = _maybe_tqdm(joined, verbose=True, desc='Loading molecular descriptors')
    return joined


def read_protein_descriptors(
    desc_type: Union[str, Descriptor, Transform] = 'unirep',
    version: VersionArg = 'latest',
    chunksize: Optional[int] = None,
    source_path: Optional[str] = None,
    ids: Optional[List[str]] = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read protein descriptors.

    :param desc_type: ``'unirep'``, ``'custom'``, or a ProDEC
        :class:`~prodec.Descriptor` / :class:`~prodec.Transform`
    :param version: dataset version to read (ignored for ``'custom'``)
    :param chunksize: rows per chunk (ignored for ProDEC)
    :param source_path: for ``'unirep'``/ProDEC: root directory for Papyrus
        data.  For ``'custom'``: path to a TSV file.
    :param ids: target identifiers to retain; ``None`` keeps all
    :param verbose: show a progress bar when reading chunks
    :param kwargs: extra keyword arguments forwarded to ProDEC ``pandas_get``
    """
    if desc_type == 'custom':
        if not os.path.isfile(source_path):
            raise ValueError(
                'source_path must point to an existing file when desc_type="custom"'
            )
        return _read_custom_protein_descriptors(source_path, chunksize, ids, verbose)

    if isinstance(desc_type, (Descriptor, Transform)):
        pv = _resolve_version(version, source_path)
        protein_data = read_protein_set(
            source_path=papyrus_version_module(pv).base.as_posix(), version=pv,
        )
        protein_data.rename(columns={'TARGET_NAME': 'target_id'}, inplace=True)
        if ids is not None:
            protein_data = protein_data[protein_data['target_id'].isin(ids)]
        protein_data = protein_data.loc[
            protein_data['Sequence'].map(desc_type.Descriptor.is_sequence_valid), :
        ]
        descriptors = desc_type.pandas_get(
            protein_data['Sequence'].tolist(),
            protein_data['target_id'].tolist(),
            **kwargs,
        )
        descriptors.rename(columns={'ID': 'target_id'}, inplace=True)
        return descriptors

    if desc_type == 'unirep':
        pv         = _resolve_version(version, source_path)
        source_mod = papyrus_version_module(pv)
        dtypes     = _load_dtypes(source_mod)
        unirep_files = locate_file(
            source_mod.join('descriptors').as_posix(),
            r'(?:\d+\.\d+_combined_prot_embeddings_unirep\.tsv.*)'
            r'|(?:\d+\.\d+_combined_protdescs_unirep\.tsv.*)',
        )
        return _read_unirep(unirep_files[0], dtypes=dtypes, chunksize=chunksize, ids=ids, verbose=verbose)

    raise ValueError(
        f'desc_type must be "unirep", "custom", or a ProDEC Descriptor/Transform, '
        f'got {desc_type!r}'
    )


def read_molecular_structures(
    is3d: bool = False,
    version: VersionArg = 'latest',
    chunksize: Optional[int] = None,
    source_path: Optional[str] = None,
    ids: Optional[List[str]] = None,
    verbose: bool = True,
) -> DataOrChunks:
    """Read molecular structures from the Papyrus SD files.

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

    sd_files = locate_file(source_mod.join('structures').as_posix(), pattern)
    id_col   = 'InChIKey' if is3d else 'connectivity'

    if chunksize is None:
        return _read_structures_full(sd_files[0], ids, id_col, verbose)
    return _read_structures_chunked(sd_files[0], chunksize, ids, id_col, verbose)


# ---------------------------------------------------------------------------
# Internal structure readers
# ---------------------------------------------------------------------------

def _read_structures_full(
    sd_file: str,
    ids: Optional[List[str]],
    id_col: str,
    verbose: bool,
) -> pd.DataFrame:
    rows = []
    with MolSupplier(sd_file, show_progress=verbose) as supplier:
        for _, mol in supplier:
            props = mol.GetPropsAsDict()
            if ids is not None and props[id_col] not in ids:
                continue
            props['mol'] = mol
            rows.append(props)
    return pd.DataFrame(rows)


def _read_structures_chunked(
    sd_file: str,
    chunksize: int,
    ids: Optional[List[str]],
    id_col: str,
    verbose: bool,
) -> Generator[pd.DataFrame, None, None]:
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError('chunksize must be a positive integer.')

    pbar_ctx = tqdm(desc='Loading molecular structures') if verbose else None

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
                    if pbar_ctx is not None:
                        pbar_ctx.update()
                    yield pd.DataFrame(rows)
                    rows = []

            if pbar_ctx is not None:
                pbar_ctx.update()
            if rows:
                yield pd.DataFrame(rows)
    finally:
        if pbar_ctx is not None:
            pbar_ctx.close()


# ---------------------------------------------------------------------------
# Internal protein-descriptor readers
# ---------------------------------------------------------------------------

def _normalise_target_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={'TARGET_NAME': 'target_id'})


def _read_unirep(
    filepath: str,
    dtypes: dict,
    chunksize: Optional[int],
    ids: Optional[List[str]],
    verbose: bool,
) -> pd.DataFrame:
    read_kw = dict(sep='\t', dtype=dtypes.get('unirep'), low_memory=True)
    chunked = chunksize is not None

    if not chunked:
        df = _normalise_target_id(pd.read_csv(filepath, **read_kw))
        return df if ids is None else df[df['target_id'].isin(ids)]

    reader = pd.read_csv(filepath, chunksize=chunksize, **read_kw)
    if verbose:
        reader = _maybe_tqdm(reader, verbose=True, desc='Loading protein descriptors')
    chunks = []
    for chunk in reader:
        chunk = _normalise_target_id(chunk)
        chunks.append(chunk if ids is None else chunk[chunk['target_id'].isin(ids)])
    return pd.concat(chunks, ignore_index=True)


def _read_custom_protein_descriptors(
    filepath: str,
    chunksize: Optional[int],
    ids: Optional[List[str]],
    verbose: bool,
) -> pd.DataFrame:
    read_kw = dict(sep='\t', low_memory=True)
    chunked = chunksize is not None

    if not chunked:
        df = _normalise_target_id(pd.read_csv(filepath, **read_kw))
        return df if ids is None else df[df['target_id'].isin(ids)]

    reader = pd.read_csv(filepath, chunksize=chunksize, **read_kw)
    if verbose:
        reader = _maybe_tqdm(reader, verbose=True, desc='Loading custom protein descriptors')
    chunks = []
    for chunk in reader:
        chunk = _normalise_target_id(chunk)
        chunks.append(chunk if ids is None else chunk[chunk['target_id'].isin(ids)])
    return pd.concat(chunks, ignore_index=True)
