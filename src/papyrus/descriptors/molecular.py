# -*- coding: utf-8 -*-

import os
import tempfile
import zipfile
import subprocess
from functools import partial
from platform import architecture
from sys import platform

import requests
import pandas as pd
from tqdm.auto import tqdm
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from e3fp.fingerprint.fprinter import Fingerprinter


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
    descs = pd.concat([data[col_mol_id],
                       descs.round(3).fillna(0)],
                      axis=1)

    return descs


def get_ecfp6(data: pd.DataFrame,
              col_mol_id: str,
              col_molecule: str,
              progress: bool = True) -> pd.DataFrame:
    """Obtain RDKit extended connectivity fingerprints with radius 3 and 2048 bits.

    :param data: A dataframe with molecules and identifiers
    :param col_mol_id: Name of the columns containing molecule identifiers
    :param col_molecule: Name of the columns containing RDKit molecules
    :param progress: Whether to show progress
    """
    # Define ECFP_6 function
    fn = partial(AllChem.GetMorganFingerprintAsBitVect, radius=3, nBits=2048)
    if progress:
        tqdm.pandas(desc='Obtaining ECFP_6')
        results = pd.concat([data[col_mol_id],
                             data[col_molecule].progress_map(fn)],
                            axis=1)
    else:
        results = pd.concat([data[col_mol_id],
                             data[col_molecule].map(fn)],
                            axis=1)
    return results


def get_e3fp(data: pd.DataFrame,
             col_mol_id: str,
             col_molecule: str,
             progress: bool = True) -> pd.DataFrame:
    """Obtain extended 3-dimensional fingerprints.

    :param data: A dataframe with molecules and identifiers
    :param col_mol_id: Name of the columns containing molecule identifiers
    :param col_molecule: Name of the columns containing RDKit molecules with 3D conformers
    :param progress: Whether to show progress
    """
    # Define helper functions
    fper = Fingerprinter(bits=2048,
                         level=5,
                         radius_multiplier=1.718,
                         stereo=True,
                         counts=False,
                         include_disconnected=True,
                         rdkit_invariants=False,
                         exclude_floating=False,
                         remove_duplicate_substructs=True)
    to_dense_fp = lambda bits, size: [1 if i in bits else 0 for i in range(size)]

    if progress:
        pbar = tqdm(data.iterrows(), total=data.shape[0], desc='Obtaining E3FP')
    else:
        pbar = data.iterrows()

    fps = []
    # Iterate over molecules
    for i, row in pbar:
        fper.run(mol=row[col_molecule])
        fp = to_dense_fp(fper.get_fingerprint_at_level().indices, 2048)
        fps.append({col_mol_id: data[col_mol_id],
                    "E3FP": ';'.join(str(x) for x in fp)})

    return pd.DataFrame(fps)


def get_mold2(data: pd.DataFrame,
              col_mol_id: str,
              col_molecule: str) -> pd.DataFrame:
    """Obtain Mold2 descriptors.

    :param data: A dataframe with molecules and identifiers
    :param col_mol_id: Name of the columns containing molecule identifiers
    :param col_molecule: Name of the columns containing RDKit molecules with 3D conformers
    """
    mold2_folder = os.path.abspath(os.path.join(__file__, os.pardir, 'extras'))
    if not os.path.isdir(os.path.join(mold2_folder, 'Mold2')):
        # Download Mold2
        session = requests.session()
        res = session.get("https://www.fda.gov/files/science%20&%20research/published/Mold2-Executable-File.zip",
                          headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
                                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                 "Chrome/39.0.2171.95 "
                                                 "Safari/537.36"},
                          stream=True, verify=True)
        zip_path = os.path.abspath(os.path.join(mold2_folder, 'Mold2-Executable-File.zip'))
        # Save ZIP file
        with open(zip_path, 'wb') as fh:
            for chunk in res.iter_content(chunk_size=1024):
                fh.write(chunk)
        # Extract Zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(mold2_folder)
        # Rename files
        os.rename(os.path.join(mold2_folder, 'Mold2', 'Windows', 'Mold2.doc'),
                  os.path.join(mold2_folder, 'Mold2', 'Windows', 'Mold2.exe'))
        os.rename(os.path.join(mold2_folder, 'Mold2', 'Windows', 'Mold2.txt'),
                  os.path.join(mold2_folder, 'Mold2', 'Windows', 'Mold2.bat'))

    # Determine components for the command
    if platform.startswith('win32'):
        exec_path = os.path.join(mold2_folder, 'Mold2', 'Windows', 'Mold2.exe')
        log_file = 'NUL'
        echo_cmd = 'echo.'
    elif platform.startswith('linux'):
        log_file = '/dev/null'
        echo_cmd = 'echo -e \'\n\''
        if architecture()[0].startswith('32'):
            exec_path = os.path.join(mold2_folder, 'Mold2', 'Linux_x86-32', 'Mold2')
        else:
            exec_path = os.path.join(mold2_folder, 'Mold2', 'Linux_x86-64', 'Mold2')
    else:
        raise RuntimeError(f'Platform ({platform}) not supported.')

    with tempfile.NamedTemporaryFile() as sdf, tempfile.NamedTemporaryFile() as output:
        # Write SD file (input of Mold2)
        with AllChem.SDWriter(sdf) as writer:
            writer.SetProps([])
            for mol in data[col_molecule]:
                writer.write(mol)
        # Create command
        command = f'{echo_cmd} | {exec_path} -i {sdf.name} -o {output.name} -r {log_file}'
        # Run calculation
        with open(os.devnull, 'wb') as devnull:
            _ = subprocess.check_output(command, shell=True, stderr=devnull)  # noqa: S602
        # Read results
        data = pd.read_table(output).drop(columns=['Number'])
    # Process output
    data = pd.concat([data[col_mol_id], data],
                     axis=1)
    return data
