# -*- coding: utf-8 -*-

"""Papyrus++ datset creation."""

import os
import lzma
from functools import partial
from collections import defaultdict
from typing import Optional
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Optional dependencies
try:
    import pystow
except ImportError as e:
    pystow = e

from papyrus_scripts.preprocess import keep_quality, equalize_cell_size_in_row, process_groups
from papyrus_scripts.reader import read_papyrus
from papyrus_scripts.utils.IO import process_data_version, get_num_rows_in_file


if isinstance(pystow, ImportError):
    raise ImportError('\nSome required dependencies are missing:\n\tpystow')


def main(root_folder: Optional[str],
         out_folder: str,
         is3d: bool = False,
         version: str = 'latest',
         xc50_tolerance: float = 0.5
         ):
    if not os.path.isdir(out_folder):
        raise ValueError('out folder does not exist')

    out_folder = os.path.abspath(out_folder)
    # Determine default paths
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    version = process_data_version(version, root_folder)
    # Create the high quality subset
    create_papyrus_pp(xc50_tolerance=xc50_tolerance, version=version, is3d=is3d,
                      root_folder=root_folder, out_folder=out_folder)


def create_papyrus_pp(outfile: Optional[str] = '++_combined_set',
                      xc50_tolerance: float = 0.5,
                      version: str = 'latest',
                      is3d: bool = False,
                      chunksize: int = 100_000,
                      root_folder: Optional[str] = None,
                      out_folder: Optional[str] = None,
                      force: bool = False,
                      njobs: int = 1) -> None:
    """Create the high quality dataset Papyrus++.

        Ki and KD values are kept untouched while IC50 and EC50 are filtered.

        :param outfile: name of the output file to be prefixed by the version and suffixed
                        by the flavour of stereochemistry used.
        :param xc50_tolerance: log units around the median to accept IC50 and EC50 data points
        :param version: version of the Papyrus dataset
        :param is3d: should the lower quality stereochemistry be considered
        :param chunksize: number of rows to be processed per process
        :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
        :param force: force the overwriting of output files (even temporary)
        :param njobs: number of simultaneous processes
        """
    # Check output folder exists
    if not os.path.isdir(os.path.abspath(os.path.join(outfile, os.pardir))):
        raise ValueError('out folder does not exist')

    # Determine default paths
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    version = process_data_version(version, root_folder)

    # Get number of chunks
    n_chunks = -(-get_num_rows_in_file('bioactivities', is3D=is3d,
                                       version=version, plusplus=False,
                                       root_folder=root_folder) // chunksize)

    # Read bioactivity data
    data = read_papyrus(is3d=is3d, version=version, plusplus=False, chunksize=chunksize, source_path=root_folder)
    # Pre-filter by high quality
    data = keep_quality(data, 'high')

    # Determine output file
    fname = os.path.join(out_folder,
                         f'{version}{outfile}_with{"" if is3d else "out"}_stereochemistry.tsv'
                         )
    if os.path.isfile(f'{fname}.xz') and not force:
        raise IOError(f'file {fname} already exists')
    elif os.path.isfile(f'{fname}.xz') and force:
        os.remove(f'{fname}.xz')
        if os.path.isfile(fname):
            os.remove(fname)

    # Process the data
    m = Manager()
    lock = m.Lock()
    pool = Pool(processes=njobs)
    list(tqdm(pool.imap(partial(process_chunk,
                                xc50_tolerance=xc50_tolerance,
                                lock=lock,
                                outfile=fname),
                        data),
              total=n_chunks,
              ncols=100,
              smoothing=0,
              desc='Processing chunks'))
    pool.close()

    # Read file and compress
    data = pd.read_csv(fname, sep='\t').sort_values('Activity_ID').reset_index(drop=True)
    with lzma.open(f'{fname}.xz', 'wt', preset=9 | lzma.PRESET_EXTREME, newline='\n') as oh:
        for i, chunk in enumerate(tqdm(np.array_split(data, 100), desc='Writing to disk', ncols=100, smoothing=0)):
            if not chunk.empty:
                chunk.to_csv(oh, sep='\t', index=False, line_terminator='\n', header=(i==0), mode='w' if i == 0 else 'a')
    os.remove(fname)


def process_chunk(chunk, xc50_tolerance, lock, outfile):
    if chunk.empty:
        return pd.DataFrame()
    types = ['IC50', 'EC50', 'KD', 'Ki']
    # Transform activity_types to column names
    activity_types = [f"type_{types[i]}" for i in range(len(types))]
    # Columns with optional multiple values
    cols2split = ['source', 'CID', 'AID', 'all_doc_ids', 'all_years', 'type_IC50', 'type_EC50', 'type_KD', 'type_Ki', 'relation',
                  'pchembl_value']
    # Keep trace of order of columns
    ordered_columns = chunk.columns.tolist()
    # Keep columns and index
    included = chunk[[x for x in chunk.columns if x in cols2split + ['Activity_ID']]]
    excluded = chunk[[x for x in chunk.columns if
                      x not in cols2split and not x.startswith('pchembl_value_') and x not in (
                      'Activity_class', 'type_other')]]
    # Split values
    included = (included.set_index('Activity_ID')  # Allows unnesting data without messing with Activity_ID
                .astype(str)  # Required for following split
                .apply(lambda x: x.str.split(';'))  # Split multiple values into lists
                .apply(equalize_cell_size_in_row, axis=1)  # Set same length of lists in each row
                .apply(pd.Series.explode)  # Unnest the data
                .reset_index())  # Recover Activity_ID
    included = included.apply(pd.to_numeric, errors='ignore')
    # Pivot type
    if 'all_doc_ids' in included.columns:
        id_vars = ['Activity_ID', 'source', 'CID', 'AID', 'all_doc_ids', 'all_years', 'pchembl_value', 'relation']
    else:
        id_vars = ['Activity_ID', 'source', 'CID', 'AID', 'pchembl_value', 'relation']
    included = included.melt(id_vars=id_vars,
                            value_vars=['type_IC50', 'type_EC50', 'type_Ki', 'type_KD'], var_name='type')
    # Keep types where value is 1
    included = included[included.value == 1].drop(columns='value')
    # Keep Ki and KD intact
    kx_included = included[included.type.isin(['type_Ki', 'type_KD'])]
    # Keep remaining IC50 and EC50
    included = included[included.type.isin(['type_EC50', 'type_IC50'])]
    kept_values = []
    # First pass : identify assays with activity values within tolerance of median
    assays_to_keep = defaultdict(lambda: [0, 0])
    for name, group in included.groupby(['Activity_ID', 'type']):
        if group.shape[0] == 1:
            continue
        median = group.pchembl_value.median()
        group = group[group.pchembl_value.between(median - xc50_tolerance, median + xc50_tolerance)]
        if not group.empty:
            for aid in group.AID.unique():
                assays_to_keep[aid][0] += 1
                assays_to_keep[aid][1] += 1
            kept_values.append(group.reset_index(drop=True))
        else:
            for aid in group.AID.unique():
                assays_to_keep[aid][1] += 1
    # Keep unique assay IDs
    assays_to_keep = [key for key, value in assays_to_keep.items() if value[0] / value[1] > 0.75]
    # Second pass: identify values of assay tagged to be kept
    for name, group in included[included.AID.isin(assays_to_keep)].groupby(['Activity_ID', 'type']):
        if group.shape[0] == 1:
            kept_values.append(group.reset_index(drop=True))
    # Aggregate groups
    included = pd.concat(kept_values).reset_index(drop=True)
    # Add kept Ki and KD values
    processed = pd.concat([included, kx_included]).reset_index(drop=True)
    # Pivot to get assay types as in original
    processed['type_var'] = 1
    processed = pd.concat([processed.drop(columns=['type', 'type_var']),
                           processed.pivot(columns=['type'], values='type_var').fillna(0).convert_dtypes()],
                          axis=1)
    # Create temporary type_other
    processed['type_other'] = ''
    # Remove group names from groupby
    _, processed = list(zip(*processed.groupby('Activity_ID')))
    # Calculate aggregated values
    if 'all_doc_ids' in processed[0].columns:
        add_cols = ['all_years', 'all_doc_ids']
    else:
        add_cols = None
    filtered = (pd.concat(list(map(partial(process_groups, additional_columns=add_cols),
                                   (processed[i:i + 1000] for i in range(0, len(processed), 1000))
                                  )))
                .reset_index(drop=True))
    filtered['Activity_class'] = ''
    # Add excluded columns
    chunk = filtered.fillna(0).merge(excluded, how='inner', on='Activity_ID')[ordered_columns]
    # Drop superfluous columns
    if 'Activity_class' in chunk.columns:
        chunk['Activity_class'] = np.NaN
    if 'type_other' in chunk.columns:
        chunk['type_other'] = np.NaN

    # Acquire lock to write to file
    with lock:
        if os.path.isfile(outfile):
            # Append to existing file
            chunk.to_csv(outfile, sep='\t', index=False, mode='a', header=False)
        else:
            # Create file
            chunk.to_csv(outfile, sep='\t', index=False, mode='w', header=True)
