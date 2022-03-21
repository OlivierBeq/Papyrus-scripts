# -*- coding: utf-8 -*-

import glob
import json
import os
from typing import Optional, Union, Iterator

import pystow
import pandas as pd

from .utils.IO import TypeDecoder


def read_papyrus(is3d: bool = False, chunksize: Optional[int] = None, source_path: Optional[str] = None) -> Union[
    Iterator[pd.DataFrame], pd.DataFrame]:
    """Read the Papyrus dataset.

    :param is3d: whether to consider stereochemistry or not (default: False)
    :param chunksize: number of lines per chunk. To read without chunks, set to None
    :param source_path: folder containing the bioactivity dataset (default: pystow's home folder)
    :return: the Papyrus activity dataset
    """
    # Determine default paths
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(source_path)
    source_path = pystow.module('papyrus')
    source_path = source_path.base.as_posix()
    # Load data types
    dtype_file = source_path.join(name='data_types.json').as_posix()
    with open(dtype_file, 'r') as jsonfile:
        dtypes = json.load(jsonfile, cls=TypeDecoder)['papyrus']
    # Find the file
    file_mask = os.path.join(source_path, f'*.*_combined_set_with{"out" if not is3d else ""}_stereochemistry.tsv*')
    filename = glob.glob(file_mask)
    # Handle WSL ZoneIdentifier files
    filename = [fname for fname in filename if not fname.endswith(':ZoneIdentifier')]
    if len(filename) == 0:
        raise FileNotFoundError('Could not find Papyrus dataset')
    return pd.read_csv(filename[0], sep='\t', chunksize=chunksize, dtype=dtypes, low_memory=True)


def read_protein_set(source_path: Optional[str] = None) -> pd.DataFrame:
    """Read the protein targets of the Papyrus dataset.

        :param source_path: folder containing the molecular descriptor datasets
        :return: the set of protein targets in the Papyrus dataset
        """
    # Determine default paths
    if source_path is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(source_path)
    source_path = pystow.module('papyrus')
    source_path = source_path.base.as_posix()
    # Find the file
    file_mask = os.path.join(source_path, f'*.*_combined_set_protein_targets.tsv*')
    filename = glob.glob(file_mask)
    # Handle WSL ZoneIdentifier files
    filename = [fname for fname in filename if not fname.endswith(':ZoneIdentifier')]
    if len(filename) == 0:
        raise FileNotFoundError('Could not find Papyrus dataset of protein targets')
    return pd.read_csv(filename[0], sep='\t', keep_default_na=False)
