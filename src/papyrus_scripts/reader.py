# -*- coding: utf-8 -*-

import json
import os
from typing import Optional, Union, Iterator, List
from functools import partial

import pystow
import pandas as pd
from tqdm.auto import tqdm
from prodec import Descriptor, Transform

from .utils.IO import locate_file, process_data_version, TypeDecoder


def read_papyrus(is3d: bool = False, version: str = 'latest', chunksize: Optional[int] = None, source_path: Optional[str] = None) -> Union[
    Iterator[pd.DataFrame], pd.DataFrame]:
    """Read the Papyrus dataset.

    :param is3d: whether to consider stereochemistry or not (default: False)
    :param version: version of the dataset to be read
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
                            f'*.*_combined_set_with{"out" if not is3d else ""}_stereochemistry.tsv*')
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
    filenames = locate_file(source_path.base.as_posix(), f'*.*_combined_set_protein_targets.tsv*')
    return pd.read_csv(filenames[0], sep='\t', keep_default_na=False)


def read_molecular_descriptors(desc_type: str = 'mold2', is3d: bool = False,
                               version: str = 'latest', chunksize: Optional[int] = None,
                               source_path: Optional[str] = None):
    """Get molecular descriptors

    :param desc_type: type of descriptor {'mold2', 'mordred', 'cddd', 'fingerprint', 'all'}
    :param is3d: whether to load descriptors of the dataset containing stereochemistry
    :param version: version of the dataset to be read
    :param chunksize: number of lines per chunk. To read without chunks, set to None
    :param source_path: folder containing the molecular descriptor datasets
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
                                  f'*.*_combined_{3 if is3d else 2}D_moldescs_mold2.tsv*')
    elif desc_type in ['mordred', 'all']:
        mordd_files = locate_file(source_path.join('descriptors').as_posix(),
                                  f'*.*_combined_{3 if is3d else 2}D_moldescs_mordred{3 if is3d else 2}D.tsv*')
    elif desc_type in ['cddd', 'all']:
        cddds_files = locate_file(source_path.join('descriptors').as_posix(),
                                  f'*.*_combined_{3 if is3d else 2}D_moldescs_CDDDs.tsv*')
    elif desc_type in ['fingerprint', 'all']:
        molfp_files = locate_file(source_path.join('descriptors').as_posix(),
                                  f'*.*_combined_{3 if is3d else 2}D_moldescs_{"E3FP" if is3d else "ECFP6"}.tsv*')
    elif desc_type in ['moe', 'all']:
        moe_files = locate_file(source_path.join('descriptors').as_posix(),
                                f'*.*_combined_{3 if is3d else 2}D_moldescs_MOE.tsv*')
    if desc_type == 'mold2':
        return pd.read_csv(mold2_files[0], sep='\t', dtype=dtypes['mold2'], low_memory=True, chunksize=chunksize)
    elif desc_type == 'mordred':
        return pd.read_csv(mordd_files[0], sep='\t', dtype=dtypes[f'mordred_{3 if is3d else 2}D'], low_memory=True,
                           chunksize=chunksize)
    elif desc_type == 'cddd':
        return pd.read_csv(cddds_files[0], sep='\t', dtype=dtypes['CDDD'], low_memory=True, chunksize=chunksize)
    elif desc_type == 'fingerprint':
        return pd.read_csv(molfp_files[0], sep='\t', dtype=dtypes[f'{"E3FP" if is3d else "ECFP6"}'], low_memory=True,
                           chunksize=chunksize)
    elif desc_type == 'moe':
        return pd.read_csv(moe_files[0], sep='\t', low_memory=True, chunksize=chunksize)
    elif desc_type == 'all':
        mold2 = pd.read_csv(mold2_files[0], sep='\t', dtype=dtypes['mold2'], low_memory=True, chunksize=chunksize)
        mordd = pd.read_csv(mordd_files[0], sep='\t', dtype=dtypes[f'mordred_{3 if is3d else 2}D'], low_memory=True,
                            chunksize=chunksize)
        cddds = pd.read_csv(cddds_files[0], sep='\t', dtype=dtypes['CDDD'], low_memory=True, chunksize=chunksize)
        molfp = pd.read_csv(molfp_files[0], sep='\t', dtype=dtypes[f'{"E3FP" if is3d else "ECFP6"}'], low_memory=True,
                            chunksize=chunksize)
        moe = pd.read_csv(moe_files[0], sep='\t', low_memory=True, chunksize=chunksize)
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
        return _join_molecular_descriptors(mold2, mordd, molfp, cddds, moe,
                                           on='InChIKey' if is3d else 'connectivity')


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



def read_protein_descriptors(desc_type: Union[str, Descriptor, Transform] = 'unirep',
                             version: str = 'latest', chunksize: Optional[int] = None,
                             source_path: Optional[str] = None,
                             ids: Optional[List[str]] = None, verbose: bool = True):
    """Get protein descriptors

   :param desc_type: type of descriptor {'unirep'} or a prodec Descriptor or Transform
   :param version: version of the dataset to be read
   :param chunksize: number of lines per chunk. To read without chunks, set to None
   :param source_path: If desc_type is 'unirep', folder containing the protein descriptor datasets.
                       If desc_type is 'custom', the file path to a dataframe containing target_id
                       as its first column and custom descriptors in the following ones.
                       If desc_type is a ProDEC Descriptor or Transform, folder containing Papyrus protein data.
   :param ids: identifiers of the sequences which descriptors should be loaded (e.g. P30542_WT)
   :param verbose: whether to show progress
   :return: the dataframe of protein descriptors
    """
    if desc_type not in ['unirep', 'custom'] and not isinstance(desc_type, (Descriptor, Transform)):
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
    # Set verbose level
    if verbose:
        pbar = partial(tqdm, desc='Loading protein descriptors')
    else:
        pbar = partial(iter)
    if desc_type == 'unirep':
        unirep_files = locate_file(source_path.join('descriptors').as_posix(), f'*.*_combined_prot_embeddings_unirep.tsv*')
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
    elif desc_type == 'custom':
        if not os.path.isfile(source_path):
            raise ValueError('source_path must be a file if using a custom descriptor type')
        if chunksize is None and ids is None:
            return pd.read_csv(source_path, sep='\t', low_memory=True)
        elif chunksize is None and ids is not None:
            descriptors = pd.read_csv(source_path, sep='\t', low_memory=True)
            if 'target_id' in descriptors.columns:
                return descriptors[descriptors['target_id'].isin(ids)]
            return descriptors[descriptors['TARGET_NAME'].isin(ids)].rename(columns={'TARGET_NAME': 'target_id'})
        elif chunksize is not None and ids is None:
            return pd.concat([chunk
                              for chunk in pbar(pd.read_csv(source_path, sep='\t',
                                                            low_memory=True, chunksize=chunksize))
                              ]).rename(columns={'TARGET_NAME': 'target_id'})
        return pd.concat([chunk[chunk['target_id'].isin(ids)]
                          if 'target_id' in chunk.columns
                          else chunk[chunk['TARGET_NAME'].isin(ids)]
                          for chunk in pbar(pd.read_csv(source_path, sep='\t', low_memory=True, chunksize=chunksize))
                          ]).rename(columns={'TARGET_NAME': 'target_id'})
    else:
        # Calculate protein descriptors
        protein_data = read_protein_set(source_path).rename(columns={'TARGET_NAME': 'target_id'})
        protein_data = protein_data[protein_data['target_id'].isin(ids)]
        descriptors = desc_type.pandas_get(protein_data['Sequence'], protein_data['target_id'], quiet=True)
        descriptors.rename(columns={'ID': 'target_id'}, inplace=True)
        return descriptors
