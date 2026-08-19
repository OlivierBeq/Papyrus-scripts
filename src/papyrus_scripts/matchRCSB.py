# -*- coding: utf-8 -*-

"""Match data of the Papyrus dataset with that of the Protein Data Bank."""

import os
import time
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path

import pandas as pd
import polars as pl
import pystow
import requests
from pandas.io.parsers import TextFileReader as PandasTextFileReader
from rdkit import Chem
from tqdm.auto import tqdm, trange

from .utils import IO, UniprotMatch
from .utils.IO import new_session, notebook_safe_ncols
from .utils.mol_reader import suppress_rdkit_log


def papyrus_rcsb_data_root(root_folder: str | Path | None = None) -> pystow.Module:
    """Return the pystow :class:`~pystow.Module` for the RCSB on-disk folder.

    :param root_folder: folder that will contain the RCSB Protein Data Bank data
        (default: pystow's home folder)
    """
    IO._set_root_folder(root_folder)
    return pystow.module('rcsb')


def get_all_pdb_ids_with_ligands(session: requests.Session | None = None) -> list[str]:
    """Obtain all ligands from the RCSB PDB Search API.

    :param session: session to reuse for the request; a new one-off session
        (with a modern User-Agent and retries) is created if not provided
    """
    session = session if session is not None else new_session()
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    # Query for all structures where non-polymer entity count > 0
    search_query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                "operator": "greater",
                "value": 0,
            },
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,  # Ensures we get the whole archive, no pagination
        },
    }
    response = session.post(search_url, json=search_query)
    response.raise_for_status()
    # Parse the response to extract just the PDB IDs
    data = response.json()
    pdb_ids = [item["identifier"] for item in data.get("result_set", [])]
    return pdb_ids


def update_rcsb_data(root_folder: str | Path | None = None,
                     overwrite: bool = False,
                     verbose: bool = True,
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
    if (output_path.is_file() and (time.time() - output_path.stat().st_mtime) < 86400) and not overwrite:
        if verbose:
            print(f'RCSB data was obtained less than 24 hours ago: {output_path}\n'
                  f'Set overwrite=True to force the fetching of data again.',
                  )
        return pd.read_csv(output_path, sep='\t')
    # Get all ligands
    if verbose:
        print('Obtaining RCSB ligands codes')
    session = new_session()
    pdb_ids = get_all_pdb_ids_with_ligands(session)
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
        pbar = trange(0, len(pdb_ids), chunk_size, desc='Gather RCSB data', ncols=notebook_safe_ncols(100))
    else:
        pbar = range(0, len(pdb_ids), chunk_size)
    # 2. Process the IDs in chunks (RDKit warnings suppressed due to InChI-fication)
    with suppress_rdkit_log():
        for i in pbar:
            chunk = pdb_ids[i:i + chunk_size]
            current_chunk = (i // chunk_size) + 1
            # Make the GraphQL request for the current chunk
            response = session.post(
                url,
                json={"query": query, "variables": {"pdbIds": chunk}},
            )
            if response.status_code != 200:
                message = (f"WARNING:\tFailed to fetch batch {current_chunk}/{total_chunks}. "
                          f"Status Code: {response.status_code}")
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
                                "InChI_3D": Chem.MolToInchi(mol),
                                # 2D InChI for 2D data
                                "InChI_2D": Chem.MolToInchi(mol_2D),
                                "PDBID_ligand": ligand_id,
                                "PDBID_protein": protein_id,
                                "SMILES": smiles_stereo,
                            },
                            )
                # 4. Respect API rate limits (Wait 0.5 seconds between requests)
                if len(pdb_ids) > chunk_size:
                    time.sleep(0.5)
        if verbose:
            pbar.close()
        # To DataFrame - explicit columns so an empty *results* (e.g. every
        # batch failed) still has PDBID_protein etc. to merge/rename below.
        results_df = pd.DataFrame.from_records(
            results, columns=['InChI_3D', 'InChI_2D', 'PDBID_ligand', 'PDBID_protein', 'SMILES'],
        )
    # Map PDBID prot to UniProt acessions
    if verbose:
        print('Obtaining mappings from protein PDB ID to UniProt accessions')
    uniprot_mapping = UniprotMatch.uniprot_mappings(results_df.PDBID_protein.unique().tolist(),
                                                    map_from='PDB',
                                                    map_to='UniProtKB_AC-ID',
                                                    )  # Forces the use of SIFTS
    # Join on the RCSB data
    if verbose:
        print('Combining RCSB and UniProt data')
    pdb_data = results_df.merge(uniprot_mapping, left_on='PDBID_protein', right_on='PDB')
    # Rename columns
    pdb_data = pdb_data.rename(columns={'UniProtKB_AC-ID': 'UniProt_accession'})
    # Drop duplicate information
    pdb_data = pdb_data.drop(columns='PDB')
    # Reorder columns
    pdb_data = pdb_data[['InChI_3D', 'InChI_2D', 'PDBID_ligand', 'SMILES', 'PDBID_protein', 'UniProt_accession']]
    # Write to disk and return
    if verbose:
        print('Writing results to disk')
    tmp_path = output_path.with_name(f'{output_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp')
    for stale in output_path.parent.glob(f'{output_path.name}.*.tmp'):
        stale.unlink(missing_ok=True)
    try:
        # explicit compression: tmp_path's name doesn't end in '.xz'
        pdb_data.to_csv(tmp_path, sep='\t', index=False, compression='xz')
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return pdb_data


def get_matches(data: pd.DataFrame | pl.DataFrame | pl.LazyFrame | PandasTextFileReader | Iterator,
                root_folder: str | Path | None = None,
                verbose: bool = True,
                total: int | None = None,
                update: bool = True) -> pd.DataFrame | Generator:
    """Match Papyrus bioactivity data to RCSB PDB identifiers.

    :param data: Papyrus data to be mapped with PDB identifiers; matching
        runs natively in polars internally, but a pandas DataFrame is
        always returned
    :param root_folder: Directory where Papyrus bioactivity data is stored (default: pystow's home folder)
    :param verbose: show progress if data is and Iterator or a PandasTextFileReader
    :param total: Total number of chunks for progress display
    :param update: should the local cache of PDB identifiers be updated
    :return: The subset of Papyrus data with matching RCSB PDB identifiers
    """
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_get_matches(data, root_folder, verbose, total)
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    if isinstance(data, pl.DataFrame):
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
        IO._set_root_folder(root_folder)
        papyrus_root = pystow.module('papyrus')
        rcsb_data_path = papyrus_root.join('rcsb', name='RCSB_data.tsv.xz')
        # Read the data mapping
        rcsb_data = pl.read_csv(rcsb_data_path, separator='\t')
        if 'SMILES' in rcsb_data.columns:
            rcsb_data = rcsb_data.drop('SMILES')
        # Inner join (default) drops unmatched rows. coalesce=False keeps
        # both sides' key columns (e.g. InChI_2D), which the .drop() below
        # expects to exist.
        data = data.join(
            rcsb_data, left_on=['InChI', 'accession'], right_on=[identifier, 'UniProt_accession'],
            maintain_order='left', coalesce=False,
        )
        data = data.drop(['InChI_2D', 'InChI_3D', 'UniProt_accession'])
        other_columns = [c for c in data.columns if c not in ('Activity_ID', 'PDBID_protein')]
        data = data.group_by('Activity_ID', maintain_order=True).agg(
            pl.col('PDBID_protein').str.join(';'),
            *(pl.col(c).first() for c in other_columns),
        )
        return data.to_pandas().set_index('Activity_ID')
    else:
        raise TypeError('data can only be a pandas DataFrame, TextFileReader or an Iterator')


def _chunked_get_matches(chunks: PandasTextFileReader | Iterator, root_folder: str | Path | None, verbose: bool,
                         total: int | None) -> Generator[pd.DataFrame, None, None]:
    if verbose:
        pbar = tqdm(chunks, total=total, ncols=notebook_safe_ncols(100))
    else:
        pbar = chunks
    for chunk in pbar:
        processed_chunk = get_matches(chunk, root_folder, update=False)
        yield processed_chunk
