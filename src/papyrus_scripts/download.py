# -*- coding: utf-8 -*-

"""Download utilities of the Papyrus scripts."""

import os
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import pystow
import requests
from tqdm.auto import tqdm

from .utils.IO import (PapyrusVersion, get_disk_space, enough_disk_space,
                       assert_sha256sum, get_papyrus_links, get_latest_online_version,
                       get_downloaded_versions, read_jsonfile, write_jsonfile)

USER_AGENT = None  # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"

# Download / integrity constants
_CHUNKSIZE = 1_048_576  # 1 MB per streaming chunk
_RETRIES = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_pystow_home(outdir: Optional[Union[str, Path]]) -> None:
    """Point pystow at *outdir* when it is not None."""
    if outdir is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(
            outdir if isinstance(outdir, str) else str(outdir)
        )


def _resolve_versions(version: Union[str, List[str]], files: dict) -> List[PapyrusVersion]:
    """Normalise the *version* argument into a sorted, deduplicated list of :class:`PapyrusVersion` objects.

    Accepts any combination of new-format canonical strings (``'2022.04.2'``),
    new-format alias strings (``'2022.04'``), ``'latest'``, and ``'all'``.

    :param version: raw version argument passed by the caller
    :param files: the links dictionary returned by :func:`get_papyrus_links`,
        used to enumerate available versions and to resolve ``'latest'``
    :raises ValueError: if an unrecognised version string is supplied
    """
    available_old_fmts = list(files.keys())
    latest_pv = get_latest_online_version()

    if not isinstance(version, list):
        version = [version]

    resolved: List[PapyrusVersion] = []
    for v in version:
        if v == 'latest':
            resolved.append(latest_pv)
        elif v == 'all':
            resolved.extend(PapyrusVersion(version=av) for av in available_old_fmts)
        else:
            try:
                pv = PapyrusVersion(version=v)
            except ValueError:
                valid = ['latest', 'all'] + available_old_fmts
                raise ValueError(
                    f'version must be one of [{", ".join(valid)}], got {v!r}'
                )
            if pv.version_old_fmt not in available_old_fmts:
                valid = ['latest', 'all'] + available_old_fmts
                raise ValueError(
                    f'version must be one of [{", ".join(valid)}], got {v!r}'
                )
            resolved.append(pv)

    # Deduplicate while preserving sort order (oldest first)
    seen: set = set()
    unique: List[PapyrusVersion] = []
    for pv in sorted(set(resolved)):
        if pv not in seen:
            seen.add(pv)
            unique.append(pv)
    return unique


def _file_path(papyrus_version_root: pystow.Module, ftype: str, fname: str) -> str:
    """Return the absolute path where *fname* of type *ftype* should be stored.

    The mapping follows the original layout:
    * bioactivity/protein/metadata files  → version root
    * structure files                     → ``structures/`` sub-folder
    * everything else (descriptors …)    → ``descriptors/`` sub-folder

    :param papyrus_version_root: pystow module for the specific version
    :param ftype: logical file-type key (e.g. ``'2D_papyrus'``, ``'2D_mold2'``)
    :param fname: bare filename
    """
    _ROOT_FTYPES = {
        'papyrus++', '2D_papyrus', '3D_papyrus',
        'proteins', 'data_types', 'data_size',
        'readme', 'license', 'requirements',
    }
    _STRUCTURE_FTYPES = {'2D_structures', '3D_structures'}

    if ftype in _ROOT_FTYPES:
        return papyrus_version_root.join(name=fname).as_posix()
    if ftype in _STRUCTURE_FTYPES:
        return papyrus_version_root.join('structures', name=fname).as_posix()
    return papyrus_version_root.join('descriptors', name=fname).as_posix()


def _iter_entries(ftype_data) -> List[dict]:
    """Return a list of file-entry dicts regardless of whether the raw value
    is a single dict or a list of dicts.

    :param ftype_data: value from the links JSON for a given file type
    :raises ValueError: if the value is neither a dict nor a list
    """
    if isinstance(ftype_data, dict):
        return [ftype_data]
    if isinstance(ftype_data, list):
        return ftype_data
    raise ValueError(
        f'Papyrus links file corrupted: expected dict or list, '
        f'got {type(ftype_data).__name__!r}.'
    )


def _total_size(ftypes: set, version_files: dict) -> int:
    """Sum the byte sizes of all entries in *ftypes* for a single version.

    :param ftypes: set of logical file-type keys to include
    :param version_files: the sub-dict from the links JSON for that version
    """
    total = 0
    for ftype in ftypes:
        for entry in _iter_entries(version_files[ftype]):
            total += entry['size']
    return total


def _update_versions_json(
        papyrus_root: pystow.Module,
        pv: PapyrusVersion,
        add: bool,
) -> None:
    """Add or remove *pv* from the local ``versions.json`` registry.

    :param papyrus_root: pystow module for the top-level Papyrus folder
    :param pv: the version to register or deregister
    :param add: True to add the version, False to remove it
    """
    json_file = papyrus_root.join(name='versions.json').as_posix()
    existing: list = read_jsonfile(json_file) if os.path.isfile(json_file) else []
    # Work with old-format strings for the JSON file (unchanged on disk).
    old_fmt = pv.version_old_fmt
    if add:
        updated = sorted(set(existing + [old_fmt]))
    else:
        updated = sorted(v for v in existing if v != old_fmt)
    write_jsonfile(updated, json_file)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_papyrus(outdir: Optional[str] = None,
                     version: Union[str, List[str]] = 'latest',
                     nostereo: bool = True,
                     stereo: bool = False,
                     only_pp: bool = True,
                     structures: bool = False,
                     descriptors: Optional[Union[str, List[str]]] = 'all',
                     progress: bool = True,
                     disk_margin: float = 0.10,
                     update_links: bool = True) -> None:
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
    :param update_links: Should links be updated (allows new versions to be fetched)
    """
    _set_pystow_home(outdir)

    files = get_papyrus_links(offline=not update_links)
    versions = _resolve_versions(version, files)

    if descriptors is None:
        descriptors = []
    if not isinstance(descriptors, list):
        descriptors = [descriptors]

    papyrus_root = pystow.module('papyrus')

    for pv in versions:
        old_fmt = pv.version_old_fmt
        version_files = files[old_fmt]
        papyrus_version_root = pystow.module('papyrus', old_fmt)

        # ------------------------------------------------------------------
        # Build the set of logical file-type keys to download
        # ------------------------------------------------------------------
        downloads: set = {'readme', 'requirements', 'proteins'}

        if nostereo:
            downloads.add('papyrus++')
            if not only_pp:
                downloads.add('2D_papyrus')
            elif progress:
                print(
                    '########## DISCLAIMER ##########\n'
                    'You are downloading the high-quality Papyrus++ dataset.\n'
                    'Should you want to access the entire, though of lower quality, Papyrus dataset,\n'
                    'look into additional switches of this command.\n'
                    '################################'
                )
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
            # prodec was added in a later version; skip silently for 'all'
            if 'proteins_prodec' in version_files:
                downloads.add('proteins_prodec')
            elif 'prodec' in descriptors:
                warnings.warn(f'ProDEC descriptors are not available for Papyrus version {pv}. Skipping.')

        # Drop any key that is absent from this version's link table
        downloads = {ft for ft in downloads if ft in version_files}

        # ------------------------------------------------------------------
        # Check available disk space
        # ------------------------------------------------------------------
        total = _total_size(downloads, version_files)

        if progress:
            print(
                f'Number of files to be downloaded: {len(downloads)}\n'
                f'Total size: {tqdm.format_sizeof(total)}B'
            )

        base = papyrus_version_root.base.as_posix()
        if not enough_disk_space(base, total, disk_margin):
            print(
                '########## ERROR ##########\n'
                f'Not enough disk space ({disk_margin:.0%} kept for safety)\n'
                f'Available: {tqdm.format_sizeof(get_disk_space(base))}B\n'
                f'Required:  {tqdm.format_sizeof(total)}B\n'
                '################################'
            )
            return

        # ------------------------------------------------------------------
        # Download
        # ------------------------------------------------------------------
        if progress:
            pbar = tqdm(
                total=total,
                desc=f'Downloading version {pv}',
                unit='B',
                unit_scale=True,
            )

        for ftype in downloads:
            for entry in _iter_entries(version_files[ftype]):
                dname = entry['name']
                durl = entry['url']
                dsize = entry['size']
                dhash = entry['sha256']
                fpath = _file_path(papyrus_version_root, ftype, dname)

                # Skip if already present and intact
                if os.path.isfile(fpath) and assert_sha256sum(fpath, dhash):
                    if progress:
                        pbar.update(dsize)
                    continue

                # Attempt download with up to _RETRIES tries
                success = False
                remaining = _RETRIES
                while not success and remaining > 0:
                    session = requests.session()
                    res = session.get(
                        durl,
                        headers={"User-Agent": USER_AGENT},
                        stream=True,
                        verify=True,
                    )
                    with open(fpath, 'wb') as fh:
                        for chunk in res.iter_content(chunk_size=_CHUNKSIZE):
                            fh.write(chunk)
                            if progress:
                                pbar.update(len(chunk))

                    success = assert_sha256sum(fpath, dhash)
                    if not success:
                        remaining -= 1
                        os.remove(fpath)
                        if progress:
                            msg = (
                                    f'SHA256 mismatch for {dname}. '
                                    + (f'Retrying ({remaining} left).'
                                       if remaining > 0
                                       else f'All {_RETRIES} attempts failed.')
                            )
                            pbar.write(msg)

                if not success:
                    if progress:
                        pbar.close()
                    raise IOError(f'Download failed for {dname}')

                # Extract ZIP archives in-place
                if dname.endswith('.zip'):
                    dest = os.path.dirname(fpath)
                    with zipfile.ZipFile(fpath) as zh:
                        for name in zh.namelist():
                            zh.extract(name, dest)
                    os.remove(fpath)

        if progress:
            pbar.close()

        # Register this version in the local versions.json
        _update_versions_json(papyrus_root, pv, add=True)


def remove_papyrus(
        outdir: Optional[Union[str, Path]] = None,
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
        progress: bool = True,
) -> None:
    """Remove locally downloaded Papyrus data.

    :param outdir: directory where Papyrus data is stored
        (default: pystow's directory)
    :param version: version(s) whose files should be removed; accepts the
        same values as :func:`download_papyrus`
    :param papyruspp: remove the Papyrus++ bioactivity file
    :param bioactivities: remove the full bioactivity file(s)
    :param proteins: remove the protein-targets file
    :param nostereo: consider 2D (no stereochemistry) files
    :param stereo: consider 3D (with stereochemistry) files
    :param structures: remove structure files
    :param descriptors: descriptor type(s) to remove; ``'all'`` removes every
        descriptor file
    :param other_files: remove metadata files (LICENSE, README, data_types,
        data_size)
    :param version_root: wipe the entire directory for the specified
        version(s); prompts for confirmation unless *force* is True
    :param papyrus_root: wipe all Papyrus data; prompts for confirmation
        unless *force* is True
    :param force: skip interactive confirmation prompts
    :param progress: display progress bars and status messages
    """
    _set_pystow_home(outdir)

    files = get_papyrus_links()
    versions = _resolve_versions(version, files)

    if not isinstance(descriptors, list):
        descriptors = [descriptors]

    papyrus_root_mod = pystow.module('papyrus')

    # ------------------------------------------------------------------
    # Nuclear option: wipe everything
    # ------------------------------------------------------------------
    if papyrus_root:
        if not force:
            confirmation = input(
                'Confirm the removal of ALL Papyrus data and versions (Y/N): '
            )
            if confirmation != 'Y':
                print('Removal was aborted.')
                return
        shutil.rmtree(papyrus_root_mod.base.as_posix())
        if progress:
            print('All Papyrus data was successfully removed.')
        return

    for pv in versions:
        old_fmt = pv.version_old_fmt
        version_files = files[old_fmt]
        papyrus_version_root = pystow.module('papyrus', old_fmt)

        # --------------------------------------------------------------
        # Per-version nuclear option: wipe one version folder
        # --------------------------------------------------------------
        if version_root:
            if not force:
                confirmation = input(
                    f'Confirm the removal of version {pv} of Papyrus data (Y/N): '
                )
                if confirmation != 'Y':
                    print('Removal was aborted.')
                    return
            shutil.rmtree(papyrus_version_root.base.as_posix())
            if progress:
                print(f'Version {pv} of Papyrus was successfully removed.')
            _update_versions_json(papyrus_root_mod, pv, add=False)
            return

        # --------------------------------------------------------------
        # Build the set of logical file-type keys to remove
        # --------------------------------------------------------------
        removal: set = set()

        if papyruspp:
            removal.add('papyrus++')
        if bioactivities and nostereo:
            removal.add('2D_papyrus')
        if bioactivities and stereo:
            removal.add('3D_papyrus')
        if proteins:
            removal.add('proteins')
        if structures and nostereo:
            removal.add('2D_structures')
        if structures and stereo:
            removal.add('3D_structures')
        if nostereo and ('mold2' in descriptors or 'all' in descriptors):
            removal.add('2D_mold2')
        if nostereo and ('cddd' in descriptors or 'all' in descriptors):
            removal.add('2D_cddd')
        if nostereo and ('mordred' in descriptors or 'all' in descriptors):
            removal.add('2D_mordred')
        if stereo and ('mordred' in descriptors or 'all' in descriptors):
            removal.add('3D_mordred')
        if nostereo and ('fingerprint' in descriptors or 'all' in descriptors):
            removal.add('2D_fingerprint')
        if stereo and ('fingerprint' in descriptors or 'all' in descriptors):
            removal.add('3D_fingerprint')
        if 'unirep' in descriptors or 'all' in descriptors:
            removal.add('proteins_unirep')
        if 'prodec' in descriptors or 'all' in descriptors:
            removal.add('proteins_prodec')
        if other_files:
            removal.update({'data_types', 'data_size', 'readme', 'license'})

        # Restrict to keys that actually exist in this version's link table
        removal = {ft for ft in removal if ft in version_files}

        # --------------------------------------------------------------
        # Determine which files are present on disk and calculate size
        # --------------------------------------------------------------
        total = 0
        present: List[str] = []  # ftypes confirmed to exist on disk

        for ftype in list(removal):
            ftype_data = version_files[ftype]
            # Multi-entry types (chunked archives) are handled uniformly
            all_entries_exist = True
            ftype_size = 0
            for entry in _iter_entries(ftype_data):
                fpath = _file_path(papyrus_version_root, ftype, entry['name'])
                if os.path.isfile(fpath):
                    ftype_size += entry['size']
                else:
                    all_entries_exist = False
            if all_entries_exist:
                total += ftype_size
                present.append(ftype)

        if progress:
            print(
                f'Number of files to be removed: {len(present)}\n'
                f'Total size: {tqdm.format_sizeof(total)}B'
            )

        if not present:
            return

        # --------------------------------------------------------------
        # Remove files
        # --------------------------------------------------------------
        if progress:
            pbar = tqdm(
                total=total,
                desc=f'Removing files from version {pv}',
                unit='B',
                unit_scale=True,
            )

        for ftype in present:
            for entry in _iter_entries(version_files[ftype]):
                fpath = _file_path(papyrus_version_root, ftype, entry['name'])
                if not os.path.isfile(fpath):
                    if progress:
                        pbar.update(entry['size'])
                    continue
                os.remove(fpath)
                if progress:
                    pbar.update(entry['size'])

        if progress:
            pbar.close()

        # Update the local versions.json registry if all bioactivity data
        # is gone (conservative: only deregister when the root folder is empty)
        remaining_files = [
            f for f in os.listdir(papyrus_version_root.base.as_posix())
            if not f.startswith('.')
        ]
        if not remaining_files:
            _update_versions_json(papyrus_root_mod, pv, add=False)
