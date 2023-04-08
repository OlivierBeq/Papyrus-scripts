# -*- coding: utf-8 -*-

"""Match data of the Papyrus dataset with that of the Protein Data Bank."""

import os
import time
from typing import Iterator, Generator, Optional, Union

import pystow
from rdkit import Chem
from rdkit import RDLogger
from tqdm.auto import tqdm
import pandas as pd
from pandas.io.parsers import TextFileReader as PandasTextFileReader
import requests

from .utils import UniprotMatch


def update_rcsb_data(root_folder: Optional[str] = None,
                     overwrite: bool = False,
                     verbose: bool = True
                     ) -> pd.DataFrame:
    """Update the local data of the RCSB.

    :param root_folder: Directory where Papyrus bioactivity data is stored (default: pystow's home folder)
    :param overwrite: Whether to overwrite the local file if already present
                      (default: False if the local file was downloaded today.
    :param verbose: Should logging information be printed.
    :return: The mapping between PDB and UniProt identifiers
    """
    # Define output path
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    root_folder = pystow.module('papyrus')
    output_path = root_folder.join('rcsb', name='RCSB_data.tsv.xz')
    # Check if file is too recent
    if (os.path.isfile(output_path) and (time.time() - os.path.getmtime(output_path)) < 86400) and not overwrite:
        if verbose:
            print(f'RCSB data was obtained less than 24 hours ago: {output_path}\n'
                  f'Set overwrite=True to force the fetching of data again.')
        return pd.read_csv(output_path, sep='\t')
    # Obtain the mapping InChI to PDB ligand code
    if verbose:
        print(f'Obtaining RCSB compound mappings from InChI to PDB ID')
    base_url = 'http://ligand-expo.rcsb.org/dictionaries/{}'
    request = requests.get(base_url.format('Components-inchi.ich'))
    if request.status_code != 200:
        raise IOError(f'resource could not be accessed: {request.url}')
    inchi_data = pd.DataFrame([line.split('\t')[:2] for line in request.text.splitlines()],
                              columns=['InChI', 'PDBID'])
    # Process InChI for 2D data
    if verbose:
        pbar = tqdm(enumerate(inchi_data.InChI), total=inchi_data.shape[0], desc='Converting InChIs', ncols=100)
    else:
        pbar = enumerate(inchi_data.InChI)
    RDLogger.DisableLog('rdApp.*')
    for i, inchi in pbar:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            Chem.RemoveStereochemistry(mol)
            inchi_data.loc[i, 'InChI_2D'] = Chem.MolToInchi(mol)
    RDLogger.EnableLog('rdApp.*')
    # Obtain the mapping of PDB ids ligand to proteins structures
    if verbose:
        print(f'Obtaining RCSB compound mappings from ligand PDB ID to protein PDB ID')
    request = requests.get(base_url.format('cc-to-pdb.tdd'))
    if request.status_code != 200:
        raise IOError(f'resource could not be accessed: {request.url}')
    pdbid_data = pd.DataFrame([line.split('\t')[:2] for line in request.text.splitlines()],
                              columns=['PDBIDlig', 'PDBIDprot'])
    # Merge both dataframe
    if verbose:
        print(f'Combining the data')
    pdb_data = inchi_data.merge(pdbid_data, left_on='PDBID', right_on='PDBIDlig')
    # Unmerge the data per protein PDB ID
    pdb_data.PDBIDprot = pdb_data.PDBIDprot.str.split()
    pdb_data = pdb_data.explode('PDBIDprot')
    # Map PDBID prot to UniProt acessions
    if verbose:
        print(f'Obtaining mappings from protein PDB ID to UniProt accessions')
    uniprot_mapping = UniprotMatch.uniprot_mappings(pdb_data.PDBIDprot.tolist(),
                                                    map_from='PDB',
                                                    map_to='UniProtKB_AC-ID')  # Forces the use of SIFTS
    # Join on the RCSB data
    if verbose:
        print(f'Combining the data')
    pdb_data = pdb_data.merge(uniprot_mapping, left_on='PDBIDprot', right_on='PDB')
    # Rename columns
    pdb_data = pdb_data.rename(columns={'InChI': 'InChI_3D',
                                        'PDBIDlig': 'PDBID_ligand',
                                        'PDBIDprot': 'PDBID_protein',
                                        'UniProtKB_AC-ID': 'UniProt_accession'})
    # Drop duplicate information
    pdb_data = pdb_data.drop(columns=['PDBID', 'PDB'])
    # Reorder columns
    pdb_data = pdb_data[['InChI_3D', 'InChI_2D', 'PDBID_ligand', 'PDBID_protein', 'UniProt_accession']]
    # Write to disk and return
    if verbose:
        print(f'Writing results to disk')
    pdb_data.to_csv(output_path, sep='\t', index=False)
    return pdb_data


def get_matches(data: Union[pd.DataFrame, PandasTextFileReader, Iterator],
                root_folder: Optional[str] = None,
                verbose: bool = True,
                total: Optional[int] = None,
                update: bool = True) -> Union[pd.DataFrame, Generator]:
    """

    :param data: Papyrus data to be mapped with PDB identifiers
    :param root_folder: Directory where Papyrus bioactivity data is stored (default: pystow's home folder)
    :param verbose: show progress if data is and Iterator or a PandasTextFileReader
    :param total: Total number of chunks for progress display
    :param update: should the local cache of PDB identifiers be updated
    :return: The subset of Papyrus data with matching RCSB PDB identifiers
    """
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_get_matches(data, root_folder, verbose, total)
    if isinstance(data, pd.DataFrame):
        if 'connectivity' in data.columns:
            identifier = 'InChI_2D'
        elif 'InChIKey' in data.columns:
            identifier = 'InChI_3D'
        elif 'accession' in data.columns:
            raise ValueError('data does not contain either connectivity or InChIKey data.')
        else:
            raise ValueError('data does not contain either connectivity, InChIKey or protein accession data.')
        # Update the data if possible
        if update:
            _ = update_rcsb_data(root_folder, verbose=verbose)
        # Set pystow root folder
        if root_folder is not None:
            os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
        root_folder = pystow.module('papyrus')
        rcsb_data_path = root_folder.join('rcsb', name='RCSB_data.tsv.xz')
        # Read the data mapping
        rcsb_data = pd.read_csv(rcsb_data_path, sep='\t')
        # Process InChI
        data = data[data['InChI'].isin(rcsb_data[identifier])]
        data = data.merge(rcsb_data, left_on=['InChI', 'accession'], right_on=[identifier, 'UniProt_accession'])
        data = data.drop(columns=['InChI_2D', 'InChI_3D', 'UniProt_accession'])
        data = data.groupby('Activity_ID').aggregate({column: ';'.join
                                                      if column == 'PDBID_protein'
                                                      else 'first'
                                                      for column in data.columns})
        return data
    else:
        raise TypeError('data can only be a pandas DataFrame, TextFileReader or an Iterator')


def _chunked_get_matches(chunks: Union[PandasTextFileReader, Iterator], root_folder: Optional[str], verbose: bool,
                         total: int):
    if verbose:
        pbar = tqdm(chunks, total=total, ncols=100)
    else:
        pbar = chunks
    for chunk in pbar:
        processed_chunk = get_matches(chunk, root_folder, update=False)
        yield processed_chunk
