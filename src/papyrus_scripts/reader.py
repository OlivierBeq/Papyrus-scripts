# -*- coding: utf-8 -*-

"""Reading capacities of the Papyrus-scripts."""

import json
import os
from typing import Optional, Union, Iterator, List
from functools import partial

import pystow
import pandas as pd
from tqdm.auto import tqdm
from prodec import Descriptor, Transform

from .utils.mol_reader import MolSupplier
from .utils.IO import locate_file, process_data_version, TypeDecoder


def read_papyrus(is3d: bool = False, version: str = 'latest', plusplus: bool = True, chunksize: Optional[int] = None, source_path: Optional[str] = None) -> Union[
    Iterator[pd.DataFrame], pd.DataFrame]:
    """Read the Papyrus dataset.

    :param is3d: whether to consider stereochemistry or not (default: False)
    :param version: version of the dataset to be read
    :param plusplus: read the Papyrus++ curated subset of very high quality
    :param chunksize: number of lines per chunk. To read without chunks, set to None
    :param source_path: folder containing the bioactivity dataset (default: pystow's home folder)
    :return: the Papyrus activity dataset
    """
    # Determine default paths
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(source_path)
    version = process_data_version(version=version, root_folder=source_path)
    source_path = pystow.module('papyrus', version)
    # Load data types
    dtype_file = source_path.join(name='data_types.json').as_posix()
    with open(dtype_file, 'r') as jsonfile:
        dtypes = json.load(jsonfile, cls=TypeDecoder)['papyrus']
    # Find the file
    filenames = locate_file(source_path.base.as_posix(),
                            r'\d+\.\d+' + (r'\+\+' if plusplus else '') + '_combined_set_'
                            f'with{"out" if not is3d else ""}' + r'_stereochemistry\.tsv.*')
    return pd.read_csv(filenames[0], sep='\t', chunksize=chunksize, dtype=dtypes, low_memory=True)


def read_protein_set(source_path: Optional[str] = None, version: str = 'latest') -> pd.DataFrame:
    """Read the protein targets of the Papyrus dataset.

        :param source_path: folder containing the molecular descriptor datasets
        :param version: version of the dataset to be read
        :return: the set of protein targets in the Papyrus dataset
        """
    version = process_data_version(version=version, root_folder=source_path)
    # Determine default paths
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(source_path)
    source_path = pystow.module('papyrus', version)
    # Find the file
    filenames = locate_file(source_path.base.as_posix(), r'\d+\.\d+_combined_set_protein_targets\.tsv.*')
    return pd.read_csv(filenames[0], sep='\t', keep_default_na=False)


def read_molecular_descriptors(desc_type: str = 'mold2', is3d: bool = False,
                               version: str = 'latest', chunksize: Optional[int] = None,
                               source_path: Optional[str] = None,
                               ids: Optional[List[str]] = None, verbose: bool = True):
    """Get molecular descriptors

    :param desc_type: type of descriptor {'mold2', 'mordred', 'cddd', 'fingerprint', 'moe', 'all'}
    :param is3d: whether to load descriptors of the dataset containing stereochemistry
    :param version: version of the dataset to be read
    :param chunksize: number of lines per chunk. To read without chunks, set to None
    :param source_path: folder containing the bioactivity dataset (default: pystow's home folder)
    :param ids: identifiers of the molecules which descriptors should be loaded
                if is3d=True, then identifiers are InChIKeys, otherwise connectivities
    :param verbose: whether to show progress
    :return: the dataframe of molecular descriptors
    """
    if desc_type not in ['mold2', 'mordred', 'cddd', 'fingerprint', 'moe', 'all']:
        raise ValueError("descriptor type not supported")
    # Determine default paths
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(source_path)
    version = process_data_version(version=version, root_folder=source_path)
    source_path = pystow.module('papyrus', version)
    # Load data types
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


def _filter_molecular_descriptors(data: Union[pd.DataFrame, Iterator],
                                  ids: Optional[List[str]], id_name: str):
    if isinstance(data, pd.DataFrame):
        if ids is None:
            return _iterate_filter_descriptors(data, None, None)
        return data[data[id_name].isin(ids)]
    else:
        return _iterate_filter_descriptors(data, ids, id_name)


def _iterate_filter_descriptors(data: Iterator, ids: Optional[List[str]], id_name: Optional[str]):
    for chunk in data:
        if ids is None:
            yield chunk
        else:
            yield chunk[chunk[id_name].isin(ids)]


def read_protein_descriptors(desc_type: Union[str, Descriptor, Transform] = 'unirep',
                             version: str = 'latest', chunksize: Optional[int] = None,
                             source_path: Optional[str] = None,
                             ids: Optional[List[str]] = None, verbose: bool = True,
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
        source_path = pystow.module('papyrus', version)
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


def read_molecular_structures(is3d: bool = False, version: str = 'latest',
                              chunksize: Optional[int] = None, source_path: Optional[str] = None,
                              ids: Optional[List[str]] = None, verbose: bool = True):
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
    source_path = pystow.module('papyrus', version)
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
                         ids: Optional[List[str]] = None,
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
