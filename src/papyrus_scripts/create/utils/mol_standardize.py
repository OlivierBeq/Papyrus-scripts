# -*- coding: utf-8 -*-

import warnings
from typing import Optional, Union

from rdkit import Chem
from papyrus_structure_pipeline import standardize as Papyrus_standardize


def standardize(mol: Union[Chem.Mol, str]) -> Optional[str]:
    """Standardize a molecule from either a RDKit molecule or SMILES.

    :param mol: RDKit molecule or SMILES of molecule to be standardized
    :return: the standardized SMILES if the molecule can be parsed
    """
    if not isinstance(mol, Chem.Mol):
        mol = Chem.MolFromSmiles(mol)
    if mol is not None:
        return Papyrus_standardize(mol, raise_error=False)
