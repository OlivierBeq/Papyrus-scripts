# -*- coding: utf-8 -*-

"""Download utilities of the Papyrus scripts."""

import os
import zipfile
import shutil
from typing import List, Optional, Union

import requests
import pystow
from tqdm.auto import tqdm

from .utils.IO import (get_disk_space, enough_disk_space, assert_sha256sum,
                       read_jsonfile, write_jsonfile, get_papyrus_links)


def download_papyrus(outdir: Optional[str] = None,
                     version: Union[str, List[str]] = 'latest',
                     nostereo: bool = True,
                     stereo: bool = False,
                     only_pp: bool = True,
                     structures: bool = False,
                     descriptors: Optional[Union[str, List[str]]] = 'all',
                     progress: bool = True,
                     disk_margin: float = 0.10) -> None:
    """Download the Papyrus data.
    
    :param outdir: directory where Papyrus data is stored (default: pystow's directory)
    :param version: version of the dataset to be downloaded
    :param nostereo: should 2D data be downloaded
    :param only_pp: download only the curated Papyrus++ subset
    :param stereo: should 3D data be downloaded
    :param structures: should molecule structures be downloaded
    :param descriptors: should molecular and protein descriptors be downloaded
    :param progress: should progress be displayed
    :param disk_margin: percent of free disk space to keep
    """

    # Determine download parameters
    CHUNKSIZE = 1048576  # 1 MB
    RETRIES = 3
    # Obtain links to files
    files = get_papyrus_links()
    available_versions = list(files.keys())
    if isinstance(version, list):
        for _version in version:
            if _version not in available_versions + ['latest', 'all']:
                raise ValueError(f'version can only be one of [{", ".join(["latest"] + available_versions)}]')
    # Identify version
    latest_version = sorted(available_versions, key=lambda s: [int(u) for u in s.split('.')])[-1]
    if version == 'latest':
        version = latest_version
        if progress:
            print(f'Latest version: {version}')
    elif isinstance(version, list) and 'latest' in version:
        for i in range(len(version)):
            if version[i] == 'latest':
                version[i] = latest_version
    elif version == 'all' or (isinstance(version, list) and 'all' in version):
        version = available_versions
    # Transform to list
    if not isinstance(version, list):
        version = [version]
    if not isinstance(descriptors, list):
        descriptors = [descriptors]
    # Remove duplicates of versions
    version = sorted(set(version), key=lambda s: [int(u) for u in s.split('.')])
    # Define root dir for downloads
    if outdir is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(outdir)
    papyrus_root = pystow.module('papyrus')
    for _version in version:
        papyrus_version_root = pystow.module('papyrus', _version)
        # Prepare files to be downloaded
        downloads = set()
        downloads.add('requirements')
        downloads.add('proteins')
        if nostereo:
            downloads.add('papyrus++')
            if not only_pp:
                downloads.add('2D_papyrus')
            elif progress:
                # Ensure this warning is printed when donwloading the Papyrus++ dataset with progress on
                print('########## DISCLAIMER ##########\n'
                      'You are downloading the high-quality Papyrus++ dataset.\n'
                      'Should you want to access the entire, though of lower quality, Papyrus dataset,\n'
                      'look into additional switches of this command.\n'
                      '################################')
            if structures:
                downloads.add('2D_structures')
            if 'mold2' in descriptors or 'all' in descriptors:
                downloads.add('2D_mold2')
            if 'cddd' in descriptors or 'all' in descriptors:
                downloads.add('2D_cddd')
            if 'mordred' in descriptors or 'all' in descriptors:
                downloads.add('2D_mordred')
            if 'fingerprint' in descriptors or 'all' in descriptors:
                downloads.add('2D_fingerprint')
        if stereo:
            downloads.add('3D_papyrus')
            if structures:
                downloads.add('3D_structures')
            if 'mordred' in descriptors or 'all' in descriptors:
                downloads.add('3D_mordred')
            if 'fingerprint' in descriptors or 'all' in descriptors:
                downloads.add('3D_fingerprint')
        if 'unirep' in descriptors or 'all' in descriptors:
            downloads.add('proteins_unirep')
        if 'prodec' in descriptors or 'all' in descriptors:
            downloads.add('proteins_prodec')
        # Determine total download size
        total = 0
        for ftype in downloads:
            if ftype == 'proteins_prodec' and ftype not in files[_version] and 'all' in descriptors:
                continue
            if isinstance(files[_version][ftype], dict):
                total += files[_version][ftype]['size']
            elif isinstance(files[_version][ftype], list):
                for subfile in files[_version][ftype]:
                    total += subfile['size']
            else:
                raise ValueError('########## ERROR ##########\n'
                                 f'Papyrus versioning file corrupted: {files[_version][ftype]} '
                                 'is neither a dict or a list.\nThis is most likely due to bad formatting '
                                 'of the underlying parsed JSON files. If you are not the maintainer, please '
                                 'remove the Papyrus data and enforce root folder removal and download '
                                 'the data before trying again.\n'
                                 '################################')
        if progress:
            print(f'Number of files to be downloaded: {len(downloads)}\n'
                  f'Total size: {tqdm.format_sizeof(total)}B')
        # Verify enough disk space
        if not enough_disk_space(papyrus_version_root.base.as_posix(), total, disk_margin):
            print('########## ERROR ##########\n'
                  f'Not enough disk space ({disk_margin:.0%} kept for safety)\n'
                  f'Available: {tqdm.format_sizeof(get_disk_space(papyrus_version_root.base.as_posix()))}B\n'
                  f'Required: {tqdm.format_sizeof(total)}B\n'
                  '################################')
            return
        # Download files
        if progress:
            pbar = tqdm(total=total, desc=f'Downloading version {_version}', unit='B', unit_scale=True)
        for ftype in downloads:
            if ftype == 'proteins_prodec' and 'proteins_prodec' not in files[_version]:
                if 'all' in descriptors:
                    continue
                else:
                    raise ValueError(f'ProDEC descriptors not available for Papyrus version {_version}')
            download = files[_version][ftype]
            if not isinstance(download, list):
                download = [download]
            for subfile in download:
                dname, durl, dsize, dhash = subfile['name'], subfile['url'], subfile['size'], subfile['sha256']
                # Determine path
                if ftype in ['papyrus++', '2D_papyrus', '3D_papyrus', 'proteins', 'data_types', 'data_size',
                             'readme', 'license', 'requirements']:
                    fpath = papyrus_version_root.join(name=dname).as_posix()
                elif ftype in ['2D_structures', '3D_structures']:
                    fpath = papyrus_version_root.join('structures', name=dname).as_posix()
                else:
                    fpath = papyrus_version_root.join('descriptors', name=dname).as_posix()
                # File already exists
                if os.path.isfile(fpath) and assert_sha256sum(fpath, dhash):
                    if progress:
                        pbar.update(dsize)
                    continue  # skip
                # Download file
                correct = False  # ensure file is not corrupted
                retries = RETRIES
                while not correct and retries > 0:  # Allow 3 failures
                    session = requests.session()
                    res = session.get(durl, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
                                                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                                   "Chrome/39.0.2171.95 "
                                                                   "Safari/537.36"},
                                      stream=True, verify=True)
                    with open(fpath, 'wb') as fh:
                        for chunk in res.iter_content(chunk_size=CHUNKSIZE):
                            fh.write(chunk)
                            if progress:
                                pbar.update(len(chunk))
                    correct = assert_sha256sum(fpath, dhash)
                    if not correct:
                        retries -= 1
                        if progress:
                            if retries > 0:
                                message = f'SHA256 hash unexpected for {dname}. Remaining download attempts: {retries}'
                            else:
                                message = f'SHA256 hash unexpected for {dname}. All {RETRIES} attempts failed.'
                            pbar.write(message)
                        os.remove(fpath)
                if retries == 0:
                    if progress:
                        pbar.close()
                    raise IOError(f'Download failed for {dname}')
                # Extract if ZIP file
                if dname.endswith('.zip'):
                    with zipfile.ZipFile(fpath) as zip_handle:
                        for name in zip_handle.namelist():
                            subpath = os.path.join(fpath, os.path.pardir)
                            zip_handle.extract(name, subpath)
                    os.remove(fpath)
        if progress:
            pbar.close()
        # Save version number
        json_file = papyrus_root.join(name='versions.json').as_posix()
        if os.path.isfile(json_file):
            data = read_jsonfile(json_file)
            data.append(_version)
            data = sorted(set(data))
            write_jsonfile(data, json_file)
        else:
            write_jsonfile([_version], json_file)


def remove_papyrus(outdir: Optional[str] = None,
                   version: Union[str, List[str]] = 'latest',
                   papyruspp: bool = False,
                   bioactivities: bool = False,
                   proteins: bool = False,
                   nostereo: bool = True,
                   stereo: bool = False,
                   structures: bool = False,
                   descriptors: Union[str, List[str]] = 'all',
                   other_files: bool = False,
                   version_root: bool = False,
                   papyrus_root: bool = False,
                   force: bool = False,
                   progress: bool = True) -> None:
    """Remove the Papyrus data.

    :param outdir: directory where Papyrus data is stored (default: pystow's directory)
    :param version: version of the dataset to be removed
    :param papyruspp: should Papyrus++ bioactivities be removed
    :param bioactivities: should bioactivity data be removed
    :param proteins:  should protein data be removed
    :param nostereo: should the files related to 2D data be considered
    :param stereo: should the files related to 3D data be considered
    :param structures: should molecule structures be removed
    :param descriptors: should molecular and protein descriptors be removed
    :param other_files: should other files (e.g. LICENSE, README, data_types, data_size) be removed
    :param version_root: remove the specified version of the papyrus data, requires confirmation
    :param papyrus_root: remove all versions of the papyrus data, requires confirmation
    :param force: disable confirmation prompt
    :param progress: should progress be displayed
    """
    # Obtain links to files
    files = get_papyrus_links()
    # Handle exceptions
    available_versions = list(files.keys())
    if isinstance(version, list):
        for _version in version:
            if _version not in available_versions + ['latest', 'all']:
                raise ValueError(f'version can only be one of [{", ".join(["latest"] + available_versions)}]')
    # Identify version
    latest_version = sorted(available_versions, key=lambda s: [int(u) for u in s.split('.')])[-1]
    if version == 'latest':
        version = latest_version
        if progress:
            print(f'Latest version: {version}')
    elif isinstance(version, list) and 'latest' in version:
        for i in range(len(version)):
            if version[i] == 'latest':
                version[i] = latest_version
    elif version == 'all' or (isinstance(version, list) and 'all' in version):
        version = available_versions
    # Transform to list
    if not isinstance(version, list):
        version = [version]
    if not isinstance(descriptors, list):
        descriptors = [descriptors]
    # Remove duplicates of versions
    version = sorted(set(version), key=lambda s: [int(u) for u in s.split('.')])
    # Define root dir for removal
    if outdir is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(outdir)
    papyrus_root_dir = pystow.module('papyrus')
    # Deep cleaning
    if papyrus_root:
        if not force:
            confirmation = input('Confirm the removal of all Papyrus data and versions (Y/N): ')
            if confirmation != 'Y':
                print('Removal was aborted.')
                return
        # Either forced or confirmed
        shutil.rmtree(papyrus_root_dir.base.as_posix())
        if progress:
            print('All Papyrus data was successfully removed.')
        return
    for _version in version:
        papyrus_version_root = pystow.module('papyrus', _version)
        # If removal of the whole version
        if version_root:
            if not force:
                confirmation = input(f'Confirm the removal of version {_version} of Papyrus data (Y/N): ')
                if confirmation != 'Y':
                    print('Removal was aborted.')
                    return
            # Either forced or confirmed
            shutil.rmtree(papyrus_version_root.base.as_posix())
            if progress:
                print(f'Version {_version} of Papyrus was successfully removed.')
            return
        # Prepare files to be removed
        removal = set()
        if bioactivities and papyruspp:
            removal.add('papyrus++')
        if bioactivities and nostereo:
            removal.add('2D_papyrus')
        elif bioactivities and stereo:
            removal.add('3D_papyrus')
        if proteins:
            removal.add('proteins')
        if structures and nostereo:
            removal.add('2D_structures')
        elif structures and stereo:
            removal.add('3D_structures')
        if nostereo and ('mold2' in descriptors or 'all' in descriptors):
            removal.add('2D_mold2')
        if nostereo and ('cddd' in descriptors or 'all' in descriptors):
            removal.add('2D_cddd')
        if nostereo and ('mordred' in descriptors or 'all' in descriptors):
            removal.add('2D_mordred')
        elif stereo and ('mordred' in descriptors or 'all' in descriptors):
            removal.add('3D_mordred')
        if nostereo and ('fingerprint' in descriptors or 'all' in descriptors):
            removal.add('2D_fingerprint')
        elif stereo and 'fingerprint' in descriptors or 'all' in descriptors:
            removal.add('3D_fingerprint')
        if 'unirep' in descriptors or 'all' in descriptors:
            removal.add('proteins_unirep')
        if 'prodec' in descriptors or 'all' in descriptors:
            removal.add('proteins_prodec')
        if other_files:
            removal.add('data_types')
            removal.add('data_size')
            removal.add('readme')
            removal.add('license')
        removal = list(removal)
        # Determine total removed size
        total = 0
        for i in range(len(removal) - 1, -1, -1):
            ftype = removal[i]
            data = files[_version][ftype]
            dname, dsize = data['name'], data['size']
            # Determine path
            if ftype in ['papyrus++', '2D_papyrus', '3D_papyrus', 'proteins', 'readme']:
                fpath = papyrus_version_root.join(name=dname).as_posix()
            elif ftype in ['2D_structures', '3D_structures']:
                fpath = papyrus_version_root.join('structures', name=dname).as_posix()
            else:
                fpath = papyrus_version_root.join('descriptors', name=dname).as_posix()
            # Handle LICENSE, data_types and data_size separately
            if other_files:
                fpath = papyrus_version_root.join(name=dname).as_posix()
                # Will throw an error if these files do not exist
                # Nevertheless they should always exist
                os.remove('data_types.json')
                os.remove('data_size.json')
                os.remove('LICENSE.txt')
            # Handle other files
            if os.path.isfile(fpath):  # file exists
                total += dsize  # add size to be removed
            else:  # file does not exist
                del removal[i]
        if progress:
            print(f'Number of files to be removed: {len(removal)}\n'
                  f'Total size: {tqdm.format_sizeof(total)}B')
        # Early stop:
        if len(removal) == 0:
            return
        # Remove files
        if progress:
            pbar = tqdm(total=total, desc=f'Removing files from version {_version}', unit='B', unit_scale=True)
        for ftype in removal:
            data = files[_version][ftype]
            dname, dsize = data['name'], data['size']
            # Determine path
            if ftype in ['papyrus++', '2D_papyrus', '3D_papyrus', 'proteins', 'data_types', 'data_size', 'readme', 'license']:
                fpath = papyrus_version_root.join(name=dname).as_posix()
            elif ftype in ['2D_structures', '3D_structures']:
                fpath = papyrus_version_root.join('structures', name=dname).as_posix()
            else:
                fpath = papyrus_version_root.join('descriptors', name=dname).as_posix()
            # File does not exist
            if not os.path.isfile(fpath):
                if progress:
                    pbar.update(dsize)
                continue  # skip
            # Remove file
            os.remove(fpath)
            pbar.update(dsize)
        if progress:
            pbar.close()
        # Remove version number
        json_file = papyrus_root_dir.join(name='versions.json').as_posix()
        if os.path.isfile(json_file):
            data = read_jsonfile(json_file)
            data = [v for v in data if v != _version]
            data = sorted(set(data))
            write_jsonfile(data, json_file)
