# -*- coding: utf-8 -*-

import warnings
from typing import Optional

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from chembl_structure_pipeline import standardizer


def standardize(mol: Optional[Chem.Mol], smiles: Optional[str]) -> str:
    """Standardize a molecule from either a RDKit molecule or SMILES.

    :param mol: RDKit molecule to be standardized
    :param smiles: SMILES of molecule to be standardized
    """
    std_smiles = _apply_chembl_standardization(mol, smiles)
    std_mol = Chem.MolFromSmiles(std_smiles)
    std_tauto = _canonicalize_tautomer(std_mol)
    final_smiles = _apply_chembl_standardization(mol=std_tauto)
    return final_smiles


def _apply_chembl_standardization(mol: Optional[Chem.Mol] = None, smiles: Optional[str] = None) -> str:
    """Apply the ChEMBL structure standardization pipeline from either a RDKit molecule or SMILES.

    :param mol: RDKit molecule to be standardized
    :param smiles: SMILES of molecule to be standardized
    """
    if mol is None and smiles is None:
        raise ValueError('Either RDKit molecule or SMILES must be specified')
    if mol is not None and smiles is not None:
        warnings.warn('Both RDKit molecule and SMILES specified; SMILES will be omitted.')
    # Use the ChEMBL structure pipeline to standardize the molecule
    if mol is not None:
        standardized_smiles, excluded = Chem.MolToSmiles(standardizer.get_parent_mol(standardizer.standardize_mol(mol)))
    else:
        try:
            RDLogger.DisableLog('rdApp.*') # Disable RDKit outputs
            mol = Chem.MolFromSmiles(smiles)
            RDLogger.EnableLog('rdApp.*') # Re-enable them
            if mol is None:
                raise ValueError(f'Could not parse SMILES: {smiles}')
            standardized_smiles, excluded = Chem.MolToSmiles(standardizer.get_parent_mol(standardizer.standardize_mol(mol)))
        except:
            raise ValueError(f'Could not parse SMILES: {smiles}')
    # Deal with molecules with exclusion flags
    if excluded:
        warnings.warn(f'Molecule containing either metal or more than 7 boron atoms: {standardized_smiles}')
    # Ensure standardized SMILES can be read back
    try:
        RDLogger.DisableLog('rdApp.*')  # Disable RDKit outputs
        standardized_mol = Chem.MolFromSmiles(standardized_smiles)
        RDLogger.EnableLog('rdApp.*')  # Re-enable them
        if standardized_mol is None:
            raise ValueError(f'Could not parse standardized SMILES: {standardized_smiles}')
    except:
        raise ValueError(f'Could not parse standardized SMILES: {standardized_smiles}')
    return standardized_smiles


def _canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """Obtain canonical tautomer of the given molecule."""
    if mol is None:
        raise ValueError('Molecule object is empty')
    # Parameter of tautomer enumeration
    enumerator = TautomerEnumerator()
    enumerator.SetMaxTautomers(0)
    can_tauto = enumerator.Canonicalize(mol)
    if can_tauto is not None:
        return can_tauto
    raise ValueError(f'Could not obtain canonical tautomer: {Chem.MolToSmiles(mol)}')
