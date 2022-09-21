# -*- coding: utf-8 -*-

import pandas as pd
from tqdm.auto import tqdm

from mordred import Calculator, descriptors

def get_mordred_descriptors(include_3d: bool,
                            data: pd.DataFrame,
                            col_mol_id: str,
                            col_molecule: str,
                            quiet: bool = True,
                            ipynb: bool = False,
                            njobs: int = 1) -> pd.DataFrame:
    """Obtain Mordred molecular descriptors.

    :param include_3d: Should 3D descriptors be calculated
    :param data: A dataframe with molecules and identifiers
    :param col_mol_id: Name of the columns containing molecule identifiers
    :param col_molecule: Name of the columns containing RDKit molecules
    :param quiet: Whether to mute progress output
    :param ipynb: Whether progress bars are being displayed in a notebook
    :param njobs: Number of concurrent processes
    """
    descs: pd.DataFrame = Calculator(descriptors, ignore_3D=not include_3d).pandas(data[col_molecule],
                                                                     quiet=quiet,
                                                                     nproc=njobs,
                                                                     ipynb=ipynb)
    descs.fillna(0, inplace=True)
    descs.insert(0, col_mol_id, data[col_mol_id])

    return descs
