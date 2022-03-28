# -*- coding: utf-8 -*-

import glob
import json
import os
from typing import Optional, Union, Iterator

import pystow
import pandas as pd

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
    version = process_data_version(version=version, root_folder=source_path)
    # Determine default paths
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(source_path)
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
