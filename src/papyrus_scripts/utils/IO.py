# -*- coding: utf-8 -*-

"""IO functions."""

import glob
import hashlib
import importlib
import inspect
import json
import os
import requests
import shutil
from typing import List, Optional

import pystow


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


def read_jsonfile(json_infile: str) -> object:
    """Read in a json file and return the json object."""
    with open(json_infile) as infile:
        data = json.load(infile)
    return data


class TypeEncoder(json.JSONEncoder):
    """Custom json encoder to support types as values."""

    def default(self, obj):
        """Add support if value if a type."""
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


def get_downloaded_versions(root_folder: str) -> List[str]:
    """Identify versions of the downloaded Papyrus data

    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    """
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    version_json = pystow.join('papyrus', name='versions.json').as_posix()
    return read_jsonfile(version_json)


def get_latest_downloaded_version(root_folder: str) -> List[str]:
    """Identify the latest version of the downloaded Papyrus data

    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    """
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    version_json = pystow.join('papyrus', name='versions.json').as_posix()
    versions = read_jsonfile(version_json)
    return sorted(versions, key=lambda s: [int(u) for u in s.split('.')])[-1]

def process_data_version(version: str, root_folder: str):
    """Confirm the version is available, downloaded and convert synonyms.

    :param version: version to be confirmed and/or converted.
    :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
    """
    # Handle exceptions
    available_versions = get_downloaded_versions(root_folder) + ['latest']
    if version not in available_versions:
        raise ValueError(f'version can only be one of [{", ".join(available_versions)}]')
    elif version == 'latest':
        version = get_latest_downloaded_version(root_folder)
    return version


def locate_file(dirpath: str, glob_pattern: str):
    """Find file(s) matching the given pattern in the given directory

    :param dirpath: Path to the directory to obtain the file from
    :param glob_pattern: Pattern passed to glob.glob to locate the file(s)
    :return: a list of files matching the pattern and in the given directory
    """
    # Handle exceptions
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f'Directory does not exist: {dirpath}')
    # Find the file
    file_mask = os.path.join(dirpath, glob_pattern)
    filenames = glob.glob(file_mask)
    # Handle WSL ZoneIdentifier files
    filenames = [fname for fname in filenames if not fname.endswith(':ZoneIdentifier')]
    if len(filenames) == 0:
        raise FileNotFoundError(f'Could not locate a file in in {dirpath} matching {file_mask}')
    return filenames


def get_num_rows_in_file(filetype: str, is3D: bool, descriptor_name: Optional[str] = None, version: str = 'latest',
                         root_folder: Optional[str] = None) -> int:
        """Get the number of rows a Papyrus file has.


        :param filetype: Type of file, one of {'bioactivity', 'structure', 'descriptor'}
        :param is3D: Whether to consider the standardised (2D) or non-standardised (3D) data
        :param descriptor_name: Name of the descriptor, one of {'cddd', 'mold2', 'mordred', 'fingerprint'},
                                only considered if type='descriptor'.
        :param version: Version of Papyrus to be considered
        :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
        :return: The number of lines in the corresponding file
        """
        if not filetype in ['bioactivities', 'structures', 'descriptors']:
            raise ValueError('filetype must be one of [\'bioactivities\', \'structures\', \'descriptors\']')
        if filetype == 'descriptors' and (
                descriptor_name is None or descriptor_name not in ['cddd', 'mold2', 'mordred', 'fingerprint']):
            raise ValueError('filetype must be one of [\'cddd\', \'mold2\', \'mordred\', \'fingerprint\']')
        # Process version shortcuts
        version = process_data_version(version=version, root_folder=root_folder)
        if root_folder is not None:
            os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
        json_file = pystow.join('papyrus', version, name='data_size.json').as_posix()
        # Obtain file sizes (number of lines)
        sizes = read_jsonfile(json_file)
        if filetype == 'bioactivities':
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

def get_papyrus_links():
    """Obtain the latest links to Papyrus data files from GitHub.

    If the connection to the GitHub server is made, the
    local version of the file is updated.
    Otherwise, defaults ot the local version of the file.
    """
    url = "https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/db-links/links.json"
    local_file = os.path.join(os.path.dirname(__file__), 'links.json')
    session = requests.session()
    try:
        res = session.get(url, verify=True)
        with open(local_file, 'w') as oh:
                oh.write(res.text)
    except ConnectionError as e:
        pass
    with open(local_file) as fh:
        data = json.load(fh)
    return data
