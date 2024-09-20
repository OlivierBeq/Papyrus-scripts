# -*- coding: utf-8 -*-

"""IO functions."""

from __future__ import annotations

import glob
import hashlib
import importlib
import inspect
import json
import os
import re
from collections import namedtuple

import requests
import shutil
import lzma
import gzip
from typing import List, Optional

import pystow
import pandas as pd
from tqdm.auto import tqdm


def sha256sum(filename, blocksize=None):
    if blocksize is None:
        blocksize = 65536
    hash = hashlib.sha256()
    with open(filename, "rb") as fh:
        for block in iter(lambda: fh.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()


def assert_sha256sum(filename, sha256, blocksize=None):
    if not (isinstance(sha256, str) and len(sha256) == 64):
        raise ValueError("SHA256 must be 64 chars: {}".format(sha256))
    sha256_actual = sha256sum(filename, blocksize)
    return sha256_actual == sha256


def write_jsonfile(data: object, json_outfile: str) -> None:
    """Write a json object to a file with lazy formatting."""
    with open(json_outfile, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def read_jsonfile(json_infile: str) -> dict:
    """Read in a json file and return the json object."""
    if not os.path.isfile(json_infile):
        return {}
    with open(json_infile) as infile:
        data = json.load(infile)
    return data


class TypeEncoder(json.JSONEncoder):
    """Custom json encoder to support types as values."""

    def default(self, obj):
        """Add support if value is a type."""
        if isinstance(obj, type):
            return {'__type__': {'module': inspect.getmodule(obj).__name__,
                                 'type': obj.__name__}
                    }
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class TypeDecoder(json.JSONDecoder):
    """Custom json decoder to support types as values."""

    def __init__(self, *args, **kwargs):
        """Simple json decoder handling types as values."""
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Handle types."""
        if '__type__' not in obj:
            return obj
        module = obj['__type__']['module']
        type_ = obj['__type__']['type']
        if module == 'builtins':
            return getattr(__builtins__, type_)
        loaded_module = importlib.import_module(module)
        return getattr(loaded_module, type_)


def enough_disk_space(destination: str,
                      required: int,
                      margin: float = 0.10):
    """Check disk has enough space.

    :param destination: folder to check
    :param required: space required in bytes
    :param margin: percent of free disk space once file is written
    """
    total, _, free = shutil.disk_usage(destination)
    return free - required > margin * total


def get_disk_space(destination: str):
    """Obtain size of free disk space.

    :param destination: folder to check
    """
    _, _, free = shutil.disk_usage(destination)
    return free


def get_downloaded_versions(root_folder: str = None) -> dict:
    """Identify versions of the downloaded Papyrus data

    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    """
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    version_json = pystow.join('papyrus', name='versions.json').as_posix()
    return read_jsonfile(version_json)


def get_downloaded_papyrus_files(root_folder: str = None) -> pd.DataFrame:
    """Identify downloaded files for each version of the Papyrus data

    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    """
    # Obtain versions downloaded
    downloaded_versions = get_downloaded_versions(root_folder)
    # Obtain filenames that could have been downloaded
    files = get_papyrus_links(offline=True)
    # Keep only file names
    file_info = namedtuple('file_info', ('version', 'short_name', 'file_name'))
    files = [file_info(version, file, file_data['name'])
             for version in downloaded_versions
             for file, file_data in files[version].items()
             if file in ['papyrus++', '2D_papyrus', '3D_papyrus', '2D_structures', '3D_structures',
                         '2D_fingerprint', '3D_fingerprint', '2D_mordred', '3D_mordred',
                         '2D_cddd', '2D_mold2', 'proteins', 'proteins_unirep', 'proteins_prodec']]
    # Try to locate files
    # Uses glob to prevent maintaining a mapping of subfolders and file names
    # This does not check files have been downloaded in the right subfolders
    data = pd.DataFrame([{'version': file.version,
                          'short_name': file.short_name,
                          'downloaded': len(glob.glob(
                              os.path.join(pystow.module('papyrus', file.version).base.as_posix(), '**',
                                           file.file_name), recursive=True)) > 0}
                         for file in files])
    return data


def get_latest_downloaded_version(root_folder: str = None) -> List[str]:
    """Identify the latest version of the downloaded Papyrus data

    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    """
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    version_json = pystow.join('papyrus', name='versions.json').as_posix()
    versions = read_jsonfile(version_json)
    return sorted(versions, key=lambda s: [int(u) for u in s.split('.')])[-1]


def get_online_versions() -> List[str]:
    """Identify the versions of the Papyrus data available online

    :return: a list of the versions available
    """
    papyrus_links = get_papyrus_links()
    return sorted(papyrus_links.keys(), key=lambda s: [int(u) for u in s.split('.')]) + ['latest']


def process_data_version(version: str | PapyrusVersion, root_folder: str = None):
    """Confirm the version is available, downloaded and convert synonyms.

    :param version: version to be confirmed and/or converted.
    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    :return: version number
    :raises: IOError is the version is not available
    """
    # Check if aliases
    if not isinstance(version, PapyrusVersion):
        version = PapyrusVersion(version=version)
    # Handle exceptions
    available_versions = get_downloaded_versions(root_folder)
    if len(available_versions) == 0:
        raise IOError('Papyrus data not available (did you download it first?)')
    else:
        available_versions += ['latest']
    if version.version_old_fmt not in available_versions:
        raise ValueError(f'version can only be one of [{", ".join(available_versions)}] not {version.version_old_fmt}')
    elif version == 'latest':
        version = get_latest_downloaded_version(root_folder)
    return version


def is_local_version_available(version: str, root_folder: str = None):
    """Confirm the version is available and downloaded

    :param version: version to check the local availability.
    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    :return: True if the version is available locally, False otherwise
    """
    try:
        _ = process_data_version(version=version, root_folder=root_folder)
        return True
    except (IOError, ValueError):
        return False

def locate_file(dirpath: str, regex_pattern: str):
    """Find file(s) matching the given pattern in the given directory

    :param dirpath: Path to the directory to obtain the file from
    :param regex_pattern: Pattern used to locate the file(s)
    :return: a list of files matching the pattern and in the given directory
    """
    # Handle exceptions
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f'Directory does not exist: {dirpath}')
    # Find the file
    filenames = [os.path.join(dirpath, fname) for fname in os.listdir(dirpath) if re.search(regex_pattern, fname)]
    # Handle WSL ZoneIdentifier files
    filenames = [fname for fname in filenames if not fname.endswith(':ZoneIdentifier')]
    if len(filenames) == 0:
        raise FileNotFoundError(f'Could not locate a file in {dirpath} matching {regex_pattern}')
    return filenames


def get_num_rows_in_file(filetype: str, is3D: bool, descriptor_name: Optional[str] = None,
                         version: str | PapyrusVersion = 'latest',
                         plusplus: bool = True, root_folder: Optional[str] = None) -> int:
    """Get the number of rows a Papyrus file has.


    :param filetype: Type of file, one of {'bioactivities', 'structures', 'descriptors'}
    :param is3D: Whether to consider the standardised (2D) or non-standardised (3D) data
    :param descriptor_name: Name of the descriptor, one of {'cddd', 'mold2', 'mordred', 'fingerprint'},
                            only considered if type='descriptors'.
    :param version: Version of Papyrus to be considered
    :param plusplus: If bioactivities come from the Papyrus++ very high quality curated set,
                     only considered if type='bioactivitities'.
    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    :return: The number of lines in the corresponding file
    """
    if filetype not in ['bioactivities', 'structures', 'descriptors']:
        raise ValueError('filetype must be one of [\'bioactivities\', \'structures\', \'descriptors\']')
    if filetype == 'descriptors' and (
            descriptor_name is None or descriptor_name not in ['cddd', 'mold2', 'mordred', 'fingerprint']):
        raise ValueError('filetype must be one of [\'cddd\', \'mold2\', \'mordred\', \'fingerprint\']')
    # Process version shortcuts
    version = process_data_version(version=version, root_folder=root_folder)
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    json_file = pystow.join('papyrus', version.version_old_fmt, name='data_size.json').as_posix()
    # Obtain file sizes (number of lines)
    sizes = read_jsonfile(json_file)
    if filetype == 'bioactivities':
        if plusplus:
            if 'papyrus_++' in sizes.keys():
                return sizes['papyrus_++']
            else:
                return sizes['papyrus++']
        return sizes['papyrus_3D'] if is3D else sizes['papyrus_2D']
    elif filetype == 'structures':
        return sizes['structures_3D'] if is3D else sizes['structures_2D']
    elif filetype == 'descriptors':
        if descriptor_name == 'cddd':
            return sizes['cddd']
        elif descriptor_name == 'mold2':
            return sizes['mold2']
        elif descriptor_name == 'fingerprint':
            return sizes['E3FP'] if is3D else sizes['ECFP6']
        elif descriptor_name == 'mordred':
            return sizes['mordred_3D'] if is3D else sizes['mordred_2D']


def get_papyrus_links(offline: bool = False):
    """Obtain the latest links to Papyrus data files from GitHub.

    If the connection to the GitHub server is made, the
    local version of the file is updated.
    Otherwise, defaults ot the local version of the file.

    :param offline: do not attempt to download the latest file from GitHub
    """
    local_file = os.path.join(os.path.dirname(__file__), 'links.json')
    if not offline:
        url = "https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/db-links/links.json"
        session = requests.session()
        try:
            res = session.get(url, verify=True)
            with open(local_file, 'w') as oh:
                oh.write(res.text)
        except requests.exceptions.ConnectionError as e:
            pass
    with open(local_file) as fh:
        data = json.load(fh)
    return data


def get_papyrus_aliases(offline: bool = False):
    """Obtain the latest aliases of the Papyrus versions from GitHub.

    If the connection to the GitHub server is made, the
    local version of the file is updated.
    Otherwise, defaults ot the local version of the file.

    :param offline: do not attempt to download the latest file from GitHub
    """
    local_file = os.path.join(os.path.dirname(__file__), 'aliases.json')
    if not offline:
        url = "https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/db-links/aliases.json"
        session = requests.session()
        try:
            res = session.get(url, verify=True)
            with open(local_file, 'w') as oh:
                oh.write(res.text)
        except requests.exceptions.ConnectionError as e:
            pass
    data = pd.read_json(local_file, orient='split', dtype={'version': 'str', 'alias': 'str',
                                                           'revision': 'str', 'chembl_version': 'str'})
    return data


def convert_xz_to_gz(input_file: str, output_file: str,
                     compression_level: int = 9,
                     progress: bool = False) -> None:
    """Convert a LZMA-compressed xz file to a GZIP-compressed file.

    :param input_file: Path of the input file
    :param output_file: Path of the output file
    :param compression_level: Compression level of the output file (if None, defaults to 9)
    :param progress: Show conversion progress.
    """
    if compression_level is None:
        compression_level = 9
    # Transform per chunk
    chunksize = 10 * 1048576  # 10 MB
    with lzma.open(input_file, 'rb') as fh, gzip.open(output_file, 'wb', compresslevel=compression_level) as oh:
        if progress:
            pbar = tqdm(desc='Determining size', unit='B', unit_scale=True)
            size = fh.seek(0, 2)  # Determine original size
            _ = fh.seek(0, 0)  # Go back to the beginning
            pbar.set_description('Converting')
            pbar.total = size
            # pbar = tqdm(total=size, desc='Converting', unit='B', unit_scale=True)
        while True:
            chunk = fh.read(chunksize)
            if not chunk:
                if progress:
                    pbar.close()
                break
            written = oh.write(chunk)
            if progress:
                pbar.update(written)


def convert_gz_to_xz(input_file: str, output_file: str,
                     compression_level: int = lzma.PRESET_DEFAULT,
                     extreme: bool = False,
                     progress: bool = False) -> None:
    """Convert a GZIP- compressed file to a LZMA-compressed xz file.

    :param input_file: Path of the input file
    :param output_file: Path of the output file
    :param compression_level: Compression level of the output file (if None, defaults to 6)
    :param extreme: Should extreme compression be toggled on top of the compression level
    :param progress: Show conversion progress.
    """
    if compression_level is None:
        compression_level = lzma.PRESET_DEFAULT
    preset = compression_level | lzma.PRESET_EXTREME if extreme else compression_level
    # Transform per chunk
    chunksize = 10 * 1048576  # 10 MB
    with gzip.open(input_file, 'rb') as fh, lzma.open(output_file, 'wb', preset=preset) as oh:
        if progress:
            pbar = tqdm(desc='Determining size', unit='B', unit_scale=True)
            size = fh.seek(0, 2)  # Determine original size
            _ = fh.seek(0, 0)  # Go back to the beginning
            pbar.set_description('Converting')
            pbar.total = size
            # pbar = tqdm(total=size, desc='Converting', unit='B', unit_scale=True)
        while True:
            chunk = fh.read(chunksize)
            if not chunk:
                if progress:
                    pbar.close()
                break
            written = oh.write(chunk)
            if progress:
                pbar.update(written)


class PapyrusVersion:

    aliases = get_papyrus_aliases(offline=True)

    def __init__(self, version: Optional[str] = None, chembl_version: Optional[int] = None,
                 chembl: Optional[bool] = None, excape: Optional[bool] = None,
                 sharma: Optional[bool] = None, christmann: Optional[bool] = None,
                 klaeger: Optional[bool] = None, merget: Optional[bool] = None,
                 pickett: Optional[bool] = None):
        """Determine the Papyrus version based on provided information.

        :param version: Version number (either older '05.4', or new format '2022.04')
        :param chembl_version: Version of ChEMBL to select the Papyrus version from
        :param chembl: Whether ChEMBL is included in the Papyrus version to select
        :param excape: Whether ExCAPED-DB is included in the Papyrus version to select
        :param sharma: Whether the Sharma et al. dataset is included in the Papyrus version to select
        :param christmann: Whether the Christmann-Franck et al. dataset is included in the Papyrus version to select
        :param klaeger: Whether the Klaeger et al. dataset is included in the Papyrus version to select
        :param merget: Whether the Merget et al. dataset is included in the Papyrus version to select
        :param pickett: Whether the Pickett et al. dataset is included in the Papyrus version to select
        """
        # Determine version from the given version name
        if version is not None:
            if version.lower() == 'latest':
                query = 'alias == alias.max()'
            else:
                query = f'version == "{version}" or alias == "{version.strip()}"'
        else:
            # Determine version from sources
            query = []
            if chembl:
                query.append('chembl')
            if excape:
                query.append('excape')
            if sharma:
                query.append('sharma')
            if christmann:
                query.append('christmann')
            if klaeger:
                query.append('klaeger')
            if merget:
                query.append('merget')
            if pickett:
                query.append('pickett')
            if chembl_version:
                query.append(f'chembl_version == "{chembl_version}"')
            query = " and ".join(query)
        # Identify the aliases matching the query
        if len(query):
            subset = self.aliases.query(query)
        else:
            subset = self.aliases
        if subset.empty:
            raise ValueError('None of the Papyrus versions match the provided information.')
        elif len(subset) > 1:
            raise ValueError(f'The provided information match multiple versions:\n\n' +
                             str(subset.set_index('version')) +
                             '\n\nChoose the version that matches your requirements.')
        else:
            params = subset.squeeze().to_dict()
            for key, value in params.items():
                if key == 'version':
                    setattr(self, 'version_old_fmt', value)
                elif key == 'alias':
                    setattr(self, 'version', value)
                else:
                    setattr(self, key, value)

    def __repr__(self):
        return f'<PapyrusVersion version={self.version} / {self.version_old_fmt}, revision={self.revision}>'
