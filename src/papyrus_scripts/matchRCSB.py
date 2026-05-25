# -*- coding: utf-8 -*-

"""Match data of the Papyrus dataset with that of the Protein Data Bank."""

import os
import time
from pathlib import Path
from typing import Iterator, Generator, Optional, Union

import pandas as pd
import pystow
import requests
from pandas.io.parsers import TextFileReader as PandasTextFileReader
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm, trange

from .utils import UniprotMatch, IO


def papyrus_rcsb_data_root(root_folder: Optional[str | Path] = None) -> pystow.Module:
    """Return the pystow :class:`~pystow.Module` for the RCSB on-disk folder.

    :param root_folder: folder that will contain the RCSB Protein Data Bank data
        (default: pystow's home folder)
    """
    IO._set_root_folder(root_folder)
    return pystow.module('rcsb')


def get_all_pdb_ids_with_ligands():
    """Obtain all ligands from the RCSB PDB Search API."""
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    # Query for all structures where non-polymer entity count > 0
    search_query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                "operator": "greater",
                "value": 0
            }
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True  # Ensures we get the whole archive, no pagination
        }
    }
    response = requests.post(search_url, json=search_query)
    response.raise_for_status()
    # Parse the response to extract just the PDB IDs
    data = response.json()
    pdb_ids = [item["identifier"] for item in data.get("result_set", [])]
    return pdb_ids


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
    path = papyrus_rcsb_data_root(root_folder)
    output_path = path.join('rcsb', name='RCSB_data.tsv.xz')
    # Check if file is too recent
    if (os.path.isfile(output_path) and (time.time() - os.path.getmtime(output_path)) < 86400) and not overwrite:
        if verbose:
            print(f'RCSB data was obtained less than 24 hours ago: {output_path}\n'
                  f'Set overwrite=True to force the fetching of data again.'
                  )
        return pd.read_csv(output_path, sep='\t')
    # Get all ligands
    if verbose:
        print(f'Obtaining RCSB ligands codes')
    pdb_ids = get_all_pdb_ids_with_ligands()
    # Obtain the PDB structure code to PDB ligand code
    url = "https://data.rcsb.org/graphql"
    query = """
        query ($pdbIds: [String!]!) {
          entries(entry_ids: $pdbIds) {
            rcsb_id
            nonpolymer_entities {
              nonpolymer_comp {
                chem_comp {
                  id
                }
                pdbx_chem_comp_descriptor {
                  descriptor
                  type
                  program
                }
              }
            }
          }
        }
        """
    results = []
    chunk_size = 200  # Max entries per GraphQL query to prevent server timeouts
    total_chunks = (len(pdb_ids) - 1) // chunk_size + 1
    if verbose:
        pbar = trange(0, len(pdb_ids), chunk_size, desc='Gather RCSB data', ncols=100)
    else:
        pbar = range(0, len(pdb_ids), chunk_size)
    # Disable RDKit wrnings due to InChI-fication
    RDLogger.DisableLog('rdApp.*')
    # 2. Process the IDs in chunks
    for i in pbar:
        chunk = pdb_ids[i:i + chunk_size]
        current_chunk = (i // chunk_size) + 1
        # Make the GraphQL request for the current chunk
        response = requests.post(
            url,
            json={"query": query, "variables": {"pdbIds": chunk}}
        )
        if response.status_code != 200:
            message = f"WARNING:\tFailed to fetch batch {current_chunk}/{total_chunks}. Status Code: {response.status_code}"
            if verbose:
                pbar.write(message)
            else:
                print(message)
            continue
        data = response.json()
        entries = data.get("data", {}).get("entries", [])
        # 3. Parse the data
        if entries:
            for entry in entries:
                protein_id = entry.get("rcsb_id")
                nonpolymers = entry.get("nonpolymer_entities")
                if not nonpolymers: continue
                for entity in nonpolymers:
                    comp = entity.get("nonpolymer_comp")
                    if not comp: continue
                    ligand_id = comp.get("chem_comp", {}).get("id")
                    smiles_stereo = None
                    descriptors = comp.get("pdbx_chem_comp_descriptor")
                    if descriptors:
                        for desc in descriptors:
                            desc_type = desc.get("type")
                            program = desc.get("program")
                            if desc_type in ["SMILES_STEREO", "SMILES_CANONICAL"]:
                                smiles_stereo = desc.get("descriptor")
                                if program == "OpenEye OEToolkits" and desc_type == "SMILES_STEREO":
                                    break
                    if smiles_stereo:
                        mol = Chem.MolFromSmiles(smiles_stereo)
                        mol_2D = Chem.Mol(mol)
                        Chem.RemoveStereochemistry(mol_2D)
                        results.append({
                            "InChI_3D": Chem.MolFromInchi(mol),
                            # 2D InChI for 2D data
                            "InChI_2D": Chem.MolFromInchi(mol_2D),
                            "PDBID_ligand": ligand_id,
                            "PDBID_protein": protein_id,
                            "SMILES": smiles_stereo
                        }
                        )
            # 4. Respect API rate limits (Wait 0.5 seconds between requests)
            if len(pdb_ids) > chunk_size:
                time.sleep(0.5)
    pbar.close()
    # To DataFrame
    results = pd.DataFrame.from_records(results)
    # Restore RDKit warnings
    RDLogger.EnableLog('rdApp.*')
    # Map PDBID prot to UniProt acessions
    if verbose:
        print(f'Obtaining mappings from protein PDB ID to UniProt accessions')
    uniprot_mapping = UniprotMatch.uniprot_mappings(results.PDBID_protein.unique().tolist(),
                                                    map_from='PDB',
                                                    map_to='UniProtKB_AC-ID'
                                                    )  # Forces the use of SIFTS
    # Join on the RCSB data
    if verbose:
        print(f'Combining RCSB and UniProt data')
    pdb_data = results.merge(uniprot_mapping, left_on='PDBID_protein', right_on='PDB')
    # Rename columns
    pdb_data = pdb_data.rename(columns={'UniProtKB_AC-ID': 'UniProt_accession'})
    # Drop duplicate information
    pdb_data = pdb_data.drop(columns='PDB')
    # Reorder columns
    pdb_data = pdb_data[['InChI_3D', 'InChI_2D', 'PDBID_ligand', 'SMILES', 'PDBID_protein', 'UniProt_accession']]
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
        if 'SMILES' in rcsb_data.columns:
            rcsb_data = rcsb_data.drop(columns='SMILES')
        # Process InChI
        data = data[data['InChI'].isin(rcsb_data[identifier])]
        data = data.merge(rcsb_data, left_on=['InChI', 'accession'], right_on=[identifier, 'UniProt_accession'])
        data = data.drop(columns=['InChI_2D', 'InChI_3D', 'UniProt_accession'])
        data = data.groupby('Activity_ID').aggregate({column: ';'.join
        if column == 'PDBID_protein'
        else 'first'
                                                      for column in data.columns}
                                                     )
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
