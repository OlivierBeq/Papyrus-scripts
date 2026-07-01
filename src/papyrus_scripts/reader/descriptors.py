# -*- coding: utf-8 -*-

"""
Reading capacities of the Papyrus-scripts.

This module has been refactored to improve maintainability and extensibility.
Key changes include:
1.  Introduction of a PapyrusSource dataclass to handle data location configuration.
2.  Use of a factory pattern to read different molecular descriptor types,
    making it easy to add new ones without modifying core logic.
3.  Decoupling of file-finding from file-reading to improve testability.
4.  A generic filtering utility to reduce code duplication.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, Iterator, List, Dict, Callable

import pystow
import pandas as pd
from prodec import Descriptor, Transform
from tqdm.auto import tqdm

from ..utils.IO import locate_file, process_data_version, TypeDecoder, PapyrusVersion
from ..utils.mol_reader import MolSupplier


# NOTE: This dataclass is defined here for convenience.
# For better organization, it should be moved to a central configuration module
# like `src/papyrus_scripts/utils/IO.py`.
@dataclass
class PapyrusSource:
    """Configuration object to locate a specific Papyrus dataset version."""
    version: PapyrusVersion
    is3d: bool = False
    source_path: Optional[str] = None

    def __post_init__(self):
        """Ensure version is processed after initialization."""
        self.version = process_data_version(version=self.version, root_folder=self.source_path)
        if self.source_path:
            os.environ['PYSTOW_HOME'] = os.path.abspath(self.source_path)

    @property
    def root(self) -> pystow.Module:
        """Get the pystow module path for this source."""
        return pystow.module('papyrus', self.version.version_old_fmt)

    @property
    def dtypes(self) -> dict:
        """Load and return the data types for this version."""
        dtype_file = self.root.join(name='data_types.json').as_posix()
        with open(dtype_file, 'r') as jsonfile:
            return json.load(jsonfile, cls=TypeDecoder)


# Generic Helper Functions
def _read_tsv_chunked(file_path: str, chunksize: Optional[int], dtypes: dict, low_memory: bool = True) -> Iterator[
    pd.DataFrame]:
    """Reads a TSV file, yielding chunks."""
    return pd.read_csv(file_path, sep='\t', chunksize=chunksize, dtype=dtypes, low_memory=low_memory)


def _filter_by_ids(
        data_iterator: Union[Iterator[pd.DataFrame], pd.DataFrame],
        ids: Optional[List[str]],
        id_column: str
) -> Union[Iterator[pd.DataFrame], pd.DataFrame]:
    """
    Filters a DataFrame or a DataFrame iterator by a list of IDs.
    Returns an iterator if the input is an iterator, or a DataFrame otherwise.
    """
    if ids is None:
        return data_iterator

    id_set = set(ids)

    def _filter_generator():
        for chunk in data_iterator:
            filtered_chunk = chunk[chunk[id_column].isin(id_set)]
            if not filtered_chunk.empty:
                yield filtered_chunk

    if isinstance(data_iterator, pd.DataFrame):
        return data_iterator[data_iterator[id_column].isin(id_set)]

    return _filter_generator()


def _consume_iterator(iterator: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Consumes an iterator of DataFrames into a single DataFrame."""
    return pd.concat(list(iterator), ignore_index=True)


# Core Data Readers
def read_papyrus(
        source: PapyrusSource,
        plusplus: bool = True,
        chunksize: Optional[int] = None
) -> Union[Iterator[pd.DataFrame], pd.DataFrame]:
    """
    Read the Papyrus bioactivity dataset.

    :param source: Configuration object for the Papyrus data source.
    :param plusplus: If True, read the Papyrus++ curated subset.
    :param chunksize: Number of lines per chunk. If None, reads the entire file.
    :return: A DataFrame or an iterator of DataFrames of the Papyrus activity dataset.
    """
    if source.is3d and plusplus:
        raise ValueError('Papyrus++ is only available without stereochemistry.')

    # Locate the correct bioactivity file
    stereo_suffix = 'with' if source.is3d else 'without'
    pp_suffix = r'\+\+' if plusplus else ''
    pattern = rf'\d+\.\d+{pp_suffix}_combined_set_{stereo_suffix}_stereochemistry\.tsv.*'
    file_path = locate_file(source.root.base.as_posix(), pattern)[0]

    # Read the data
    data_iterator = _read_tsv_chunked(file_path, chunksize, source.dtypes.get('papyrus', {}))

    if chunksize is None:
        return _consume_iterator(data_iterator)
    return data_iterator


def read_protein_set(source: PapyrusSource) -> pd.DataFrame:
    """
    Read the protein targets of the Papyrus dataset.

    :param source: Configuration object for the Papyrus data source.
    :return: DataFrame of protein targets.
    """
    pattern = r'\d+\.\d+_combined_set_protein_targets\.tsv.*'
    file_path = locate_file(source.root.base.as_posix(), pattern)[0]
    return pd.read_csv(file_path, sep='\t', keep_default_na=False)


# --- Molecular Descriptors Section ---

def _get_descriptor_reader(source: PapyrusSource, name: str, dtype_key: str) -> Iterator[pd.DataFrame]:
    """Helper to locate and read a specific descriptor file."""
    pattern = rf'\d+\.\d+_combined_{3 if source.is3d else 2}D_moldescs_{name}\.tsv.*'
    file_path = locate_file(source.root.join('descriptors').as_posix(), pattern)[0]
    return _read_tsv_chunked(file_path, chunksize=None, dtypes=source.dtypes.get(dtype_key, {}))


# Factory functions for each descriptor type
_read_mold2 = partial(_get_descriptor_reader, name='mold2', dtype_key='mold2')
_read_mordred_2d = partial(_get_descriptor_reader, name='mordred2D', dtype_key='mordred_2D')
_read_mordred_3d = partial(_get_descriptor_reader, name='mordred3D', dtype_key='mordred_3D')
_read_cddd = partial(_get_descriptor_reader, name='CDDDs', dtype_key='CDDD')
_read_ecfp6 = partial(_get_descriptor_reader, name='ECFP6', dtype_key='ECFP6')
_read_e3fp = partial(_get_descriptor_reader, name='E3FP', dtype_key='E3FP')
_read_moe = partial(_get_descriptor_reader, name='MOE', dtype_key={})

# Descriptor Reader Registry
DESCRIPTOR_READERS: Dict[str, Callable[[PapyrusSource], Iterator[pd.DataFrame]]] = {
    'mold2': _read_mold2,
    'cddd': _read_cddd,
    'moe': _read_moe,
}


def _join_molecular_descriptors(*descriptors: Iterator[pd.DataFrame], on: str) -> pd.DataFrame:
    """Concatenate multiple types of molecular descriptors on the same identifier."""
    dfs = [_consume_iterator(desc).set_index(on) for desc in descriptors]
    return pd.concat(dfs, axis=1).reset_index()


def read_molecular_descriptors(
        source: PapyrusSource,
        desc_type: str,
        chunksize: Optional[int] = None,
        ids: Optional[List[str]] = None,
        verbose: bool = True
) -> Union[Iterator[pd.DataFrame], pd.DataFrame]:
    """
    Get molecular descriptors using a factory pattern.

    :param source: Configuration object for the Papyrus data source.
    :param desc_type: Type of descriptor {'mold2', 'mordred', 'cddd', 'fingerprint', 'moe', 'all'}.
    :param chunksize: Number of lines per chunk. If None, reads the entire file.
    :param ids: Identifiers of the molecules for which descriptors should be loaded.
    :param verbose: If True, show a progress bar (currently placeholder).
    :return: A DataFrame or iterator of DataFrames of molecular descriptors.
    """
    id_column = 'InChIKey' if source.is3d else 'connectivity'

    # Add 2D/3D-specific readers to the registry
    if source.is3d:
        DESCRIPTOR_READERS['mordred'] = _read_mordred_3d
        DESCRIPTOR_READERS['fingerprint'] = _read_e3fp
    else:
        DESCRIPTOR_READERS['mordred'] = _read_mordred_2d
        DESCRIPTOR_READERS['fingerprint'] = _read_ecfp6

    desc_type_lower = desc_type.lower()
    if desc_type_lower not in DESCRIPTOR_READERS and desc_type_lower != 'all':
        raise ValueError(
            f"Descriptor type '{desc_type}' not supported. Choose from {list(DESCRIPTOR_READERS.keys()) + ['all']}"
            )

    # Handle 'all' separately
    if desc_type_lower == 'all':
        all_iterators = [reader(source) for reader in DESCRIPTOR_READERS.values()]
        # Note: chunking is not supported for 'all' as it requires merging full datasets
        if chunksize is not None:
            raise ValueError("Chunking is not supported when desc_type is 'all'. Set chunksize to None.")

        full_df = _join_molecular_descriptors(*all_iterators, on=id_column)
        return _filter_by_ids(full_df, ids, id_column)

    # Get the reader function for the specified type
    reader_func = DESCRIPTOR_READERS[desc_type_lower]

    # Re-implement chunking at this level for clarity
    pattern = reader_func.keywords['name']
    dtype_key = reader_func.keywords['dtype_key']

    descriptor_file = locate_file(source.root.join('descriptors').as_posix(),
                                  rf'\d+\.\d+_combined_{3 if source.is3d else 2}D_moldescs_{pattern}\.tsv.*'
                                  )[0]

    data_iterator = _read_tsv_chunked(descriptor_file, chunksize, source.dtypes.get(dtype_key, {}))

    if verbose and chunksize:
        # Placeholder for progress bar if total can be calculated
        data_iterator = tqdm(data_iterator, desc=f'Loading {desc_type} descriptors')

    filtered_data = _filter_by_ids(data_iterator, ids, id_column)

    if chunksize is None:
        return _consume_iterator(filtered_data)

    return filtered_data


def read_protein_descriptors(
        source: PapyrusSource,
        desc_type: Union[str, Descriptor, Transform],
        chunksize: Optional[int] = None,  # Note: chunksize only applies to 'unirep' and 'custom'
        ids: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs
) -> pd.DataFrame:
    """
    Get protein descriptors.

    :param source: Configuration object for the Papyrus data source.
    :param desc_type: Type of descriptor {'unirep'}, 'custom', or a prodec instance.
    :param chunksize: Number of lines per chunk (applies to precomputed files).
    :param ids: Identifiers of the sequences for which descriptors should be loaded.
    :param verbose: If True, show progress.
    :param kwargs: Keyword arguments for ProDEC's `pandas_get` method.
    :return: DataFrame of protein descriptors.
    """
    if isinstance(desc_type, (Descriptor, Transform)):
        # Calculate descriptors on the fly with ProDEC
        protein_data = read_protein_set(source)
        protein_data.rename(columns={'TARGET_NAME': 'target_id'}, inplace=True)

        if ids:
            protein_data = protein_data[protein_data['target_id'].isin(ids)]

        protein_data = protein_data.loc[protein_data['Sequence'].map(desc_type.Descriptor.is_sequence_valid), :]

        descriptors = desc_type.pandas_get(protein_data['Sequence'].tolist(), protein_data['target_id'].tolist(),
                                           **kwargs
                                           )
        return descriptors.rename(columns={'ID': 'target_id'})

    if desc_type == 'unirep':
        pattern = r'(?:\d+\.\d+_combined_prot_embeddings_unirep\.tsv.*)|(?:\d+\.\d+_combined_protdescs_unirep\.tsv.*)'
        file_path = locate_file(source.root.join('descriptors').as_posix(), pattern)[0]
        dtypes = source.dtypes.get('unirep', {})
    elif desc_type == 'custom':
        if not source.source_path or not os.path.isfile(source.source_path):
            raise ValueError('For custom descriptors, source_path must be a valid file path.')
        file_path = source.source_path
        dtypes = {}
    else:
        raise ValueError("Descriptor type not supported.")

    data_iterator = _read_tsv_chunked(file_path, chunksize, dtypes)
    filtered_iterator = _filter_by_ids(data_iterator, ids, 'target_id')

    df = _consume_iterator(filtered_iterator)
    return df.rename(columns={'TARGET_NAME': 'target_id'})


def read_molecular_structures(
        source: PapyrusSource,
        chunksize: Optional[int] = None,
        ids: Optional[List[str]] = None,
        verbose: bool = True
) -> Union[Iterator[pd.DataFrame], pd.DataFrame]:
    """
    Get molecular structures from SD file.

    :param source: Configuration object for the Papyrus data source.
    :param chunksize: Number of molecules per chunk.
    :param ids: Identifiers of the molecules to load.
    :param verbose: If True, show a progress bar.
    :return: A DataFrame or an iterator of DataFrames with molecular structures.
    """
    stereo_suffix = f'with{"" if source.is3d else "out"}_stereochemistry'
    pattern = rf'\d+\.\d+_combined_{3 if source.is3d else 2}D_set_{stereo_suffix}\.sd.*'
    sd_file = locate_file(source.root.join('structures').as_posix(), pattern)[0]
    id_column = 'InChIKey' if source.is3d else 'connectivity'

    def _structures_generator():
        data_chunk = []
        id_set = set(ids) if ids else None

        with MolSupplier(sd_file, show_progress=verbose) as f_handle:
            for _, mol in f_handle:
                props = mol.GetPropsAsDict()

                if id_set and props.get(id_column) not in id_set:
                    continue

                props['mol'] = mol
                data_chunk.append(props)

                if chunksize and len(data_chunk) >= chunksize:
                    yield pd.DataFrame(data_chunk)
                    data_chunk = []

            if data_chunk:
                yield pd.DataFrame(data_chunk)

    iterator = _structures_generator()

    if chunksize is None:
        return _consume_iterator(iterator)

    return iterator
