# -*- coding: utf-8 -*-

import glob
import json
import os
from typing import Optional, Union, Iterator

import pandas as pd

from .utils.IO import TypeDecoder


def read_papyrus(is3d: bool = False, chunksize: Optional[int] = None, source_path: str = './') -> Union[
    Iterator[pd.DataFrame], pd.DataFrame]:
    """Read the Papyrus dataset.

    :param is3d: whether to consider stereochemistry or not (default: False)
    :param chunksize: number of lines per chunk. To read without chunks, set to None
    :param source_path: folder containing the bioactivity dataset
    :return: the Papyrus activity dataset
    """
    # Load data types
    dtype_file = os.path.join(os.path.dirname(__file__), 'utils', 'data_types.json')
    with open(dtype_file, 'r') as jsonfile:
        dtypes = json.load(jsonfile, cls=TypeDecoder)['papyrus']
    file_mask = os.path.join(source_path, f'*.*_combined_set_with{"out" if not is3d else ""}_stereochemistry.tsv*')
    filename = glob.glob(file_mask)
    if len(filename) == 0:
        raise ValueError('Could not find Papyrus dataset')
    return pd.read_csv(filename[0], sep='\t', chunksize=chunksize, dtype=dtypes, low_memory=True)


def read_protein_set(source_path: str = './') -> pd.DataFrame:
    """Read the protein targets of the Papyrus dataset.

        :param source_path: folder containing the molecular descriptor datasets
        :return: the set of protein targets in the Papyrus dataset
        """
    # Load data types
    file_mask = os.path.join(source_path, f'*.*_combined_set_protein_targets.tsv*')
    filename = glob.glob(file_mask)
    if len(filename) == 0:
        raise ValueError('Could not find Papyrus dataset of protein targets')
    return pd.read_csv(filename[0], sep='\t', keep_default_na=False)
