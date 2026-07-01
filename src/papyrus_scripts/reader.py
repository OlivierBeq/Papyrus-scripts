# -*- coding: utf-8 -*-

"""Reading capacities of the Papyrus-scripts."""
from __future__ import annotations

import json
import os
from typing import Optional, Iterator
from functools import partial

import pystow
import pandas as pd
from tqdm.auto import tqdm
from prodec import Descriptor, Transform

from .utils.mol_reader import MolSupplier
from .utils.IO import PapyrusSource, TypeDecoder


def _get_dtypes(source: PapyrusSource) -> dict:
    """Load the data types definition file for the specific source version."""
    with open(source.path_data_types(), 'r') as jsonfile:
        return json.load(jsonfile, cls=TypeDecoder)


def _read_tsv_chunked(file_path: str, chunksize: Optional[int], dtypes: dict, low_memory: bool = True) -> Iterator[
    pd.DataFrame]:
    """Reads a TSV file, yielding chunks."""
    return pd.read_csv(file_path, sep='\t', chunksize=chunksize, dtype=dtypes, low_memory=low_memory)


def _consume_iterator(iterator: Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Consumes an iterator of DataFrames into a single DataFrame."""
    return pd.concat(list(iterator), ignore_index=True)


def _filter_by_ids(
        data_iterator: Iterator[pd.DataFrame] | pd.DataFrame,
        ids: Optional[list[str]],
        id_column: str
) -> Iterator[pd.DataFrame] | pd.DataFrame:
    """
    Filters a DataFrame or a DataFrame iterator by a list of IDs.
    Returns an iterator if the input is an iterator, or a DataFrame otherwise.
    """
    if ids is None:
        return data_iterator

    id_set = set(ids)

    def _filter_generator():
        # Iterate manually to apply filter per chunk
        if isinstance(data_iterator, pd.DataFrame):
            # Should not happen given the 'if check' above, but for safety
            yield data_iterator[data_iterator[id_column].isin(id_set)]
        else:
            for chunk in data_iterator:
                filtered_chunk = chunk[chunk[id_column].isin(id_set)]
                if not filtered_chunk.empty:
                    yield filtered_chunk

    if isinstance(data_iterator, pd.DataFrame):
        return data_iterator[data_iterator[id_column].isin(id_set)]

    return _filter_generator()


def read_papyrus(
    source: PapyrusSource,
    chunksize: Optional[int] = None
) -> Iterator[pd.DataFrame] | pd.DataFrame:
    """
    Read the Papyrus bioactivity dataset.

    :param source: Configuration object for the Papyrus data source.
    :param chunksize: Number of lines per chunk. If None, reads the entire file.
    :return: A DataFrame or an iterator of DataFrames of the Papyrus activity dataset.
    """
    # 1. Get File Path via Source
    file_path = source.path_bioactivity()
    # 2. Load Dtypes
    all_dtypes = _get_dtypes(source)
    dtypes = all_dtypes.get('papyrus', {})
    # 3. Read Data
    data_iterator = _read_tsv_chunked(file_path, chunksize, dtypes)
    if chunksize is None:
        return _consume_iterator(data_iterator)
    return data_iterator



def read_protein_set(source: PapyrusSource) -> pd.DataFrame:
    """
    Read the protein targets of the Papyrus dataset.

    :param source: Configuration object for the Papyrus data source.
    :return: DataFrame of protein targets.
    """
    file_path = source.path_proteins()
    return pd.read_csv(file_path, sep='\t', keep_default_na=False)


def read_molecular_descriptors(
    source: PapyrusSource,
    desc_type: str,
    chunksize: Optional[int] = None,
    ids: Optional[list[str]] = None,
    verbose: bool = True
) -> Iterator[pd.DataFrame] | pd.DataFrame:
    """Get molecular descriptors

    :param source: Configuration object for the Papyrus data source.
    :param desc_type: Type of descriptor {'mold2', 'mordred', 'cddd', 'fingerprint', 'all'}.
    :param chunksize: number of lines per chunk. To read without chunks, set to None
    :param ids: identifiers of the molecules which descriptors should be loaded
                if source.is3d=True, then identifiers are InChIKeys, otherwise connectivities
    :param verbose: whether to show progress
    :return: the dataframe of molecular descriptors
    """
    id_column = 'InChIKey' if source.is3d else 'connectivity'
    desc_type_lower = desc_type.lower()
    # Define metadata for each descriptor type based on 2D vs 3D
    # Map: key -> (filename_identifier, dtype_json_key)
    descriptor_map = {
        'mold2': ('mold2', 'mold2'),
        'cddd': ('CDDDs', 'CDDD'),
    }
    if source.is3d:
        descriptor_map['mordred'] = ('mordred3D', 'mordred_3D')
        descriptor_map['fingerprint'] = ('E3FP', 'E3FP')
    else:
        descriptor_map['mordred'] = ('mordred2D', 'mordred_2D')
        descriptor_map['fingerprint'] = ('ECFP6', 'ECFP6')
    # Validate Input
    valid_keys = list(descriptor_map.keys()) + ['all']
    if desc_type_lower not in valid_keys:
        raise ValueError(f"Descriptor type '{desc_type}' not supported. Choose from {valid_keys}")
    # Load data types
    all_dtypes = _get_dtypes(source)
    if desc_type_lower == 'all':
        # Recursive call to load each specific type
        iterators = []
        for key in descriptor_map.keys():
            # Call self for specific type, force chunksize=None to get DataFrame,
            # wrap in iterator for uniform handling in _join
            df = read_molecular_descriptors(source, key, chunksize=None, ids=None, verbose=False)
            iterators.append(iter([df]))

        full_df = _join_molecular_descriptors(*iterators, on=id_column)
        return _filter_by_ids(full_df, ids, id_column)


    dtype_file = source_path.join(name='data_types.json').as_posix()
    with open(dtype_file, 'r') as jsonfile:
        dtypes = json.load(jsonfile, cls=TypeDecoder)
    # Find the files
    if desc_type in ['mold2', 'all']:
        mold2_files = locate_file(source_path.join('descriptors').as_posix(),
                                  rf'\d+\.\d+_combined_{3 if is3d else 2}D_moldescs_mold2\.tsv.*')
    elif desc_type in ['mordred', 'all']:
        mordd_files = locate_file(source_path.join('descriptors').as_posix(),
                                  rf'\d+\.\d+_combined_{3 if is3d else 2}D_moldescs_mordred{3 if is3d else 2}D\.tsv.*')
    elif desc_type in ['cddd', 'all']:
        cddds_files = locate_file(source_path.join('descriptors').as_posix(),
                                  rf'\d+\.\d+_combined_{3 if is3d else 2}D_moldescs_CDDDs.tsv.*')
    elif desc_type in ['fingerprint', 'all']:
        molfp_files = locate_file(source_path.join('descriptors').as_posix(),
                                  rf'\d+\.\d+_combined_{3 if is3d else 2}D_moldescs_{"E3FP" if is3d else "ECFP6"}\.tsv.*')
    elif desc_type in ['moe', 'all']:
        moe_files = locate_file(source_path.join('descriptors').as_posix(),
                                rf'\d+\.\d+_combined_{3 if is3d else 2}D_moldescs_MOE\.tsv.*')
    if verbose:
        pbar = partial(tqdm, desc='Loading molecular descriptors')
    else:
        pbar = partial(iter)
    if desc_type == 'mold2':
        return _filter_molecular_descriptors(pbar(pd.read_csv(mold2_files[0], sep='\t',
                                                              dtype=dtypes['mold2'], low_memory=True, chunksize=chunksize)),
                                                  ids, 'InChIKey' if is3d else 'connectivity')
    elif desc_type == 'mordred':
        return _filter_molecular_descriptors(pbar(pd.read_csv(mordd_files[0], sep='\t',
                                                              dtype=dtypes[f'mordred_{3 if is3d else 2}D'], low_memory=True,
                                                              chunksize=chunksize)),
                                                  ids, 'InChIKey' if is3d else 'connectivity')
    elif desc_type == 'cddd':
        return _filter_molecular_descriptors(pbar(pd.read_csv(cddds_files[0], sep='\t',
                                                              dtype=dtypes['CDDD'], low_memory=True, chunksize=chunksize)),
                                                  ids, 'InChIKey' if is3d else 'connectivity')
    elif desc_type == 'fingerprint':
        return _filter_molecular_descriptors(pbar(pd.read_csv(molfp_files[0], sep='\t',
                                                              dtype=dtypes[f'{"E3FP" if is3d else "ECFP6"}'],
                                                              low_memory=True, chunksize=chunksize)),
                                                  ids, 'InChIKey' if is3d else 'connectivity')
    elif desc_type == 'moe':
        return _filter_molecular_descriptors(pbar(pd.read_csv(moe_files[0], sep='\t',
                                                              low_memory=True, chunksize=chunksize)),
                                                  ids, 'InChIKey' if is3d else 'connectivity')
    elif desc_type == 'all':
        mold2 = _filter_molecular_descriptors(pd.read_csv(mold2_files[0], sep='\t',
                                                          dtype=dtypes['mold2'], low_memory=True, chunksize=chunksize),
                                              ids, 'InChIKey' if is3d else 'connectivity')
        mordd = _filter_molecular_descriptors(pd.read_csv(mordd_files[0], sep='\t',
                                                          dtype=dtypes[f'mordred_{3 if is3d else 2}D'],
                                                          low_memory=True, chunksize=chunksize),
                                              ids, 'InChIKey' if is3d else 'connectivity')
        cddds = _filter_molecular_descriptors(pd.read_csv(cddds_files[0], sep='\t', dtype=dtypes['CDDD'],
                                                          low_memory=True, chunksize=chunksize),
                                              ids, 'InChIKey' if is3d else 'connectivity')
        molfp = _filter_molecular_descriptors(pd.read_csv(molfp_files[0], sep='\t',
                                                          dtype=dtypes[f'{"E3FP" if is3d else "ECFP6"}'],
                                                          low_memory=True, chunksize=chunksize),
                                              ids, 'InChIKey' if is3d else 'connectivity')
        moe = _filter_molecular_descriptors(pd.read_csv(moe_files[0], sep='\t', low_memory=True, chunksize=chunksize),
                                            ids, 'InChIKey' if is3d else 'connectivity')
        if chunksize is None:
            mold2.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            mordd.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            molfp.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            cddds.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            moe.set_index('InChIKey' if is3d else 'connectivity', inplace=True)
            data = pd.concat([mold2, mordd, cddds, molfp, moe], axis=1)
            del mold2, mordd, cddds, molfp, moe
            data.reset_index(inplace=True)
            return data
        return _filter_molecular_descriptors(pbar(_join_molecular_descriptors(mold2, mordd, molfp, cddds, moe,
                                                                              on='InChIKey' if is3d else 'connectivity')),
                                                  ids, 'InChIKey' if is3d else 'connectivity')


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


def _filter_molecular_descriptors(data: pd.DataFrame | Iterator,
                                  ids: Optional[list[str]], id_name: str):
    if isinstance(data, pd.DataFrame):
        if ids is None:
            return _iterate_filter_descriptors(data, None, None)
        return data[data[id_name].isin(ids)]
    else:
        return _iterate_filter_descriptors(data, ids, id_name)


def _iterate_filter_descriptors(data: Iterator, ids: Optional[list[str]], id_name: Optional[str]):
    for chunk in data:
        if ids is None:
            yield chunk
        else:
            yield chunk[chunk[id_name].isin(ids)]


def read_protein_descriptors(desc_type: str | Descriptor | Transform = 'unirep',
                             version: str | PapyrusVersion = 'latest', chunksize: Optional[int] = None,
                             source_path: Optional[str] = None,
                             ids: Optional[list[str]] = None, verbose: bool = True,
                             **kwargs):
    """Get protein descriptors

   :param desc_type: type of descriptor {'unirep'} or a prodec instance of a Descriptor or Transform
   :param version: version of the dataset to be read
   :param chunksize: number of lines per chunk. To read without chunks, set to None
   :param source_path: If desc_type is 'unirep', folder containing the protein descriptor datasets.
   If desc_type is 'custom', the file path to a tab-separated dataframe containing target_id
   as its first column and custom descriptors in the following ones.
   If desc_type is a ProDEC Descriptor or Transform instance, folder containing the bioactivity dataset
   (default: pystow's home folder)
   :param ids: identifiers of the sequences which descriptors should be loaded (e.g. P30542_WT)
   :param verbose: whether to show progress
   :param kwargs: keyword arguments passed to the `pandas` method of the ProDEC Descriptor or Transform instance
                  (is ignored if `desc_type` is not a ProDEC Descriptor or Transform instance)
   :return: the dataframe of protein descriptors
    """
    if desc_type not in ['unirep', 'custom'] and not isinstance(desc_type, (Descriptor, Transform)):
        raise ValueError("descriptor type not supported")
    if desc_type != 'custom':
        # Determine default paths
        if source_path is not None:
            os.environ['PYSTOW_HOME'] = os.path.abspath(source_path)
        version = process_data_version(version=version, root_folder=source_path)
        source_path = pystow.module('papyrus', version.version_old_fmt)
        if not isinstance(desc_type, (Descriptor, Transform)):
            # Load data types
            dtype_file = source_path.join(name='data_types.json').as_posix()
            with open(dtype_file, 'r') as jsonfile:
                dtypes = json.load(jsonfile, cls=TypeDecoder)
            # Set verbose level
            if verbose:
                pbar = partial(tqdm, desc='Loading protein descriptors')
            else:
                pbar = partial(iter)
            if desc_type == 'unirep':
                unirep_files = locate_file(source_path.join('descriptors').as_posix(), r'(?:\d+\.\d+_combined_prot_embeddings_unirep\.tsv.*)|(?:\d+\.\d+_combined_protdescs_unirep\.tsv.*)')
                if len(unirep_files) == 0:
                    raise ValueError('Could not find unirep descriptor file')
                if desc_type == 'unirep':
                    if chunksize is None and ids is None:
                        return pd.read_csv(unirep_files[0], sep='\t', dtype=dtypes['unirep'], low_memory=True)
                    elif chunksize is None and ids is not None:
                        descriptors = pd.read_csv(unirep_files[0], sep='\t', dtype=dtypes['unirep'], low_memory=True)
                        if 'target_id' in descriptors.columns:
                            return descriptors[descriptors['target_id'].isin(ids)]
                        return descriptors[descriptors['TARGET_NAME'].isin(ids)].rename(columns={'TARGET_NAME': 'target_id'})
                    elif chunksize is not None and ids is None:
                        return pd.concat([chunk
                                          for chunk in pbar(pd.read_csv(unirep_files[0], sep='\t', dtype=dtypes['unirep'],
                                                                        low_memory=True, chunksize=chunksize))
                                          ]).rename(columns={'TARGET_NAME': 'target_id'})
                    return pd.concat([chunk[chunk['target_id'].isin(ids)]
                                      if 'target_id' in chunk.columns
                                      else chunk[chunk['TARGET_NAME'].isin(ids)]
                                      for chunk in pbar(pd.read_csv(unirep_files[0], sep='\t', dtype=dtypes['unirep'],
                                                               low_memory=True, chunksize=chunksize))
                                      ]).rename(columns={'TARGET_NAME': 'target_id'})
        else:
            # Calculate protein descriptors
            protein_data = read_protein_set(pystow.module('').base.as_posix(), version=version)
            protein_data.rename(columns={'TARGET_NAME': 'target_id'}, inplace=True)
            # Keep only selected proteins
            if ids is not None:
                protein_data = protein_data[protein_data['target_id'].isin(ids)]
            # Filter out non-natural amino-acids
            protein_data = protein_data.loc[protein_data['Sequence'].map(desc_type.Descriptor.is_sequence_valid), :]
            # Obtain descriptors
            descriptors = desc_type.pandas_get(protein_data['Sequence'].tolist(), protein_data['target_id'].tolist(),
                                               **kwargs)
            descriptors.rename(columns={'ID': 'target_id'}, inplace=True)
            return descriptors
    elif desc_type == 'custom':
        # Check path exists
        if not os.path.isfile(source_path):
            raise ValueError('source_path must point to an existing file if using a custom descriptor type')
        # No chunksier, no filtering
        if chunksize is None and ids is None:
            return pd.read_csv(source_path, sep='\t', low_memory=True).rename(columns={'TARGET_NAME': 'target_id'})
        # No chunksize but filtering
        elif chunksize is None and ids is not None:
            descriptors = pd.read_csv(source_path, sep='\t', low_memory=True)
            descriptors.rename(columns={'TARGET_NAME': 'target_id'}, inplace=True)
            return descriptors[descriptors['target_id'].isin(ids)]
        else:
            # Set verbose level
            if verbose:
                pbar = partial(tqdm, desc='Loading custom protein descriptors')
            else:
                pbar = partial(iter)
            # Chunksize but no filtering
            if chunksize is not None and ids is None:
                return pd.concat([chunk
                                  for chunk in pbar(pd.read_csv(source_path, sep='\t',
                                                                low_memory=True, chunksize=chunksize))
                                  ]).rename(columns={'TARGET_NAME': 'target_id'})
            # Both chunksize and filtering
            return pd.concat([chunk[chunk['target_id'].isin(ids)]
                              if 'target_id' in chunk.columns
                              else chunk[chunk['TARGET_NAME'].isin(ids)]
                              for chunk in pbar(pd.read_csv(source_path,
                                                            sep='\t', low_memory=True, chunksize=chunksize))
                              ]).rename(columns={'TARGET_NAME': 'target_id'})


def read_molecular_structures(is3d: bool = False, version: str | PapyrusVersion = 'latest',
                              chunksize: Optional[int] = None, source_path: Optional[str] = None,
                              ids: Optional[list[str]] = None, verbose: bool = True):
    """Get molecular structures

    :param is3d: whether to load descriptors of the dataset containing stereochemistry
    :param version: version of the dataset to be read
    :param chunksize: number of lines per chunk. To read without chunks, set to None
    :param source_path: folder containing the bioactivity dataset (default: pystow's home folder)
    :param ids: identifiers of the molecules which descriptors should be loaded
                if is3d=True, then identifiers are InChIKeys, otherwise connectivities
    :param verbose: whether to show progress
    :return: the dataframe of molecular structures
    """
    # Determine default paths
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(source_path)
    version = process_data_version(version=version, root_folder=source_path)
    source_path = pystow.module('papyrus', version.version_old_fmt)
    # Find the files
    sd_files = locate_file(source_path.join('structures').as_posix(),
                              rf'\d+\.\d+_combined_{3 if is3d else 2}D_set_with{"" if is3d else "out"}_stereochemistry.sd.*')
    if chunksize is None:
        data = []
        # Iterate through the file
        with MolSupplier(sd_files[0], show_progress=True) as f_handle:
            for _, mol in f_handle:
                # Obtain SD molecular properties
                props = mol.GetPropsAsDict()
                # If IDs given and not in the list, skip
                if ids is not None and props['InChIKey' if is3d else 'connectivity'] not in ids:
                    continue
                # Else add structure to the dict
                # and add the dict to data
                props['mol'] = mol
                data.append(props)
        # Return the list of dicts as a pandas DataFrame
        return pd.DataFrame(data)
    else:
        # Process the data through an iterator
        structure_iterator = _structures_iterator(sd_files[0], chunksize, ids, is3d, verbose)
        return structure_iterator


def _structures_iterator(sd_file: str, chunksize: int,
                         ids: Optional[list[str]] = None,
                         is3d: bool = False, verbose: bool = True) -> Iterator[pd.DataFrame]:
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError('Chunksize must be a non-null positive integer.')
    if verbose:
        pbar = tqdm(desc='Loading molecular structures')
    data = []
    # Iterate through the file
    with MolSupplier(sd_file) as f_handle:
        for _, mol in f_handle:
            # Obtain SD molecular properties
            props = mol.GetPropsAsDict()
            # If IDs given and not in the list, skip
            id_ = props['InChIKey' if is3d else 'connectivity']
            if (ids is not None) and (id_ not in ids):
                continue
            props['mol'] = mol
            data.append(props)
            # Chunk is complete
            if len(data) == chunksize:
                if verbose:
                    pbar.update()
                yield pd.DataFrame(data)
                data = []
        if verbose:
            pbar.update()
        yield pd.DataFrame(data)
