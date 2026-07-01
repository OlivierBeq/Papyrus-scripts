# -*- coding: utf-8 -*-

"""IO functions."""

from __future__ import annotations

import glob
import gzip
import hashlib
import importlib
import inspect
import json
import lzma
import os
import re
import shutil
import warnings
from collections import namedtuple
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pystow
import requests
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------

def sha256sum(filename: str, blocksize: int = 65536) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(filename, "rb") as fh:
        for block in iter(lambda: fh.read(blocksize), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_sha256sum(filename: str, sha256: str, blocksize: int = 65536) -> bool:
    """Return True if the file's SHA-256 matches *sha256*, False otherwise.

    :param filename: path to the file to check
    :param sha256: expected 64-character hex digest
    :param blocksize: read block size in bytes
    :raises ValueError: if *sha256* is not a 64-character string
    """
    if not (isinstance(sha256, str) and len(sha256) == 64):
        raise ValueError("SHA256 must be a 64-character hex string, got: {!r}".format(sha256))
    return sha256sum(filename, blocksize) == sha256


def write_jsonfile(data: object, json_outfile: str) -> None:
    """Serialise *data* to *json_outfile* with readable indentation."""
    with open(json_outfile, 'w') as fh:
        json.dump(data, fh, indent=4)


def read_jsonfile(json_infile: str) -> dict | list:
    """Deserialise *json_infile* and return the resulting object.

    Returns an empty dict if the file does not exist.
    """
    if not os.path.isfile(json_infile):
        return {}
    with open(json_infile) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# JSON encode/decode helpers that preserve Python type objects as values
# ---------------------------------------------------------------------------

class TypeEncoder(json.JSONEncoder):
    """Custom JSON encoder that serialises Python type objects as values."""

    def default(self, obj):
        """Add support if value is a type."""
        if isinstance(obj, type):
            return {
                '__type__': {
                    'module': inspect.getmodule(obj).__name__,
                    'type': obj.__name__
                }
            }
        return json.JSONEncoder.default(self, obj)


class TypeDecoder(json.JSONDecoder):
    """Custom JSON decoder that deserialises Python type objects from values."""

    def __init__(self, *args, **kwargs):
        """Simple JSON decoder handling types as values."""
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Handle types."""
        if '__type__' not in obj:
            return obj
        module = obj['__type__']['module']
        type_ = obj['__type__']['type']
        if module == 'builtins':
            return getattr(__builtins__, type_)
        return getattr(importlib.import_module(module), type_)


# ---------------------------------------------------------------------------
# Disk-space helpers
# ---------------------------------------------------------------------------

def enough_disk_space(destination: str, required: int, margin: float = 0.10) -> bool:
    """Check whether *destination*'s drive has enough headroom.

    :param destination: folder (or file path) whose drive is checked
    :param required: bytes that will be written
    :param margin: fraction of total capacity to keep free after writing
    """
    total, _, free = shutil.disk_usage(destination)
    return free - required > margin * total


def get_disk_space(destination: str) -> int:
    """Return free bytes on the drive containing *destination*.

    :param destination: folder to check
    """
    _, _, free = shutil.disk_usage(destination)
    return free


# ---------------------------------------------------------------------------
# Remote-file helpers
# ---------------------------------------------------------------------------

def get_papyrus_links(offline: bool = False) -> dict:
    """Return the mapping of Papyrus version → file metadata.

    When *offline* is False the function attempts to refresh the local
    ``links.json`` cache from GitHub before returning its contents.

    :param offline: skip the network request and use the cached file only
    """
    local_file = os.path.join(os.path.dirname(__file__), 'links.json')
    if not offline:
        url = "https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/db-links/links.json"
        try:
            response = requests.session().get(url, verify=True)
            response.raise_for_status()
            with open(local_file, 'w') as fh:
                fh.write(response.text)
        except requests.exceptions.RequestException:
            pass  # fall through to the cached copy
    with open(local_file) as fh:
        return json.load(fh)


def get_papyrus_aliases(offline: bool = False) -> pd.DataFrame:
    """Return the DataFrame of Papyrus version aliases.

    When *offline* is False the function attempts to refresh the local
    ``aliases.json`` cache from GitHub before returning its contents.

    :param offline: skip the network request and use the cached file only
    """
    local_file = os.path.join(os.path.dirname(__file__), 'aliases.json')
    if not offline:
        url = "https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/db-links/aliases.json"
        try:
            response = requests.session().get(url, verify=True)
            response.raise_for_status()
            with open(local_file, 'w') as oh:
                oh.write(response.text)
        except requests.exceptions.RequestException:
            pass  # fall through to the cached copy
    return pd.read_json(
        local_file,
        orient='split',
        dtype={
            'version': 'str',
            'alias': 'str',
            'revision': 'str',
            'chembl_version': 'str',
        },
    )


# ---------------------------------------------------------------------------
# PapyrusVersion
# ---------------------------------------------------------------------------

class PapyrusVersion:
    """Represents a specific release of the Papyrus dataset.

    A version can be constructed from a version string (old format ``'05.4'``
    or new alias format ``'2022.04'``, or the special value ``'latest'``), or
    from a combination of source-inclusion flags.

    Attributes set from the aliases table
    (``_version_old_fmt``, ``_version``, ``revision``, ``chembl``, …) are
    attached dynamically during ``__init__``.

    The canonical string representation of a version is
    ``'<alias>.<revision>'``, e.g. ``'2022.04.2'``.  Access it via the
    :attr:`version` property.  The old two-part format (e.g. ``'05.4'``) is
    available via the private attribute ``version_old_fmt`` and should only
    be used internally when constructing pystow paths.
    """

    #: Class-level alias table, loaded once from the local cache.
    aliases: pd.DataFrame = get_papyrus_aliases(offline=True)

    def __init__(
            self,
            version: Optional[str] = None,
            revision: Optional[str] = None,
            chembl_version: Optional[int] = None,
            chembl: Optional[bool] = None,
            excape: Optional[bool] = None,
            sharma: Optional[bool] = None,
            christmann: Optional[bool] = None,
            klaeger: Optional[bool] = None,
            merget: Optional[bool] = None,
            pickett: Optional[bool] = None,
    ):
        """Determine the Papyrus version based on provided information.

        :param version: version string in old (``'05.4'``) or new
            (``'2022.04'``) format, or ``'latest'``
        :param revision: revision number of the desired Papyrus version
        :param chembl_version: ChEMBL release number used to select a version
        :param chembl: whether ChEMBL is included in the desired version
        :param excape: whether ExCAPE-DB is included
        :param sharma: whether the Sharma et al. dataset is included
        :param christmann: whether the Christmann-Franck et al. dataset is included
        :param klaeger: whether the Klaeger et al. dataset is included
        :param merget: whether the Merget et al. dataset is included
        :param pickett: whether the Pickett et al. dataset is included
        """
        if version is not None:
            if version.lower() == 'latest':
                # Two-step: find the highest alias, then the highest revision within it.
                # A single query using revision.max() would compute the max over the
                # full table (not the latest-alias subset) and fail when the latest
                # alias does not have the globally highest revision number.
                latest_alias = self.aliases['alias'].max()
                latest_rev = (
                    self.aliases[self.aliases['alias'] == latest_alias]['revision'].max()
                )
                query = f'alias == "{latest_alias}" and revision == "{latest_rev}"'
            elif version.count('.') == 2:
                if revision is not None:
                    raise ValueError(
                        'Revision number provided too many times '
                        '(as `revision` and as part of `version`)'
                    )
                parts = version.split('.')
                split_version = '.'.join(parts[:2])
                split_revision = parts[2]
                # Revision is stored as a string column; quote the literal so the
                # pandas query does a string comparison instead of int comparison.
                query = (
                    f'(version == "{split_version}" or alias == "{split_version}") '
                    f'and (revision == "{split_revision}")'
                )
            elif revision is not None:
                query = (
                    f'(version == "{version}" or alias == "{version}") '
                    f'and (revision == "{revision}")'
                )
            else:
                warnings.warn('Revision number not provided; latest revision selected.')
                latest_rev = (
                    self.aliases[
                        (self.aliases['version'] == version) | (self.aliases['alias'] == version)
                    ]['revision'].max()
                )
                query = (
                    f'(version == "{version}" or alias == "{version}") '
                    f'and (revision == "{latest_rev}")'
                )
        else:
            predicates: List[str] = []
            for flag, col in [
                (chembl_version, 'chembl_version'),
                (revision, 'revision'),
            ]:
                if flag is not None:
                    predicates.append(f'{col} == "{flag}"')
            for flag, col in [
                (chembl, 'chembl'),
                (excape, 'excape'),
                (sharma, 'sharma'),
                (christmann, 'christmann'),
                (klaeger, 'klaeger'),
                (merget, 'merget'),
                (pickett, 'pickett'),
            ]:
                if flag:
                    predicates.append(col)
            query = ' and '.join(predicates)

        subset = self.aliases.query(query) if query else self.aliases

        if subset.empty:
            raise ValueError('No Papyrus version matches the supplied information.')
        if len(subset) > 1:
            raise ValueError(
                'The supplied information matches multiple versions:\n\n'
                + str(
                    subset.drop(columns='version')
                    .rename(columns={'alias': 'version'})
                    .set_index('version')
                )
                + '\n\nNarrow your criteria to select a single version.'
            )

        self.params: dict = {}
        for key, value in subset.squeeze().to_dict().items():
            if key not in ('version', 'alias', 'revision'):
                self.params[key] = value
            else:
                attr = (
                    '_version_old_fmt' if key == 'version'
                    else ('_version' if key == 'alias' else key)
                )
                setattr(self, attr, str(value))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Canonical version string ``'<alias>.<revision>'``, e.g. ``'2022.04.2'``."""
        return f'{self._version}.{self.revision}'

    @property
    def version_old_fmt(self) -> str:
        """Old-style version string (e.g. ``'05.4'``); only used internally for pystow path construction."""
        return self._version_old_fmt

    @property
    def is_latest(self) -> bool:
        """Return True if this is the most recent version in the known releases table."""
        latest_alias = self.aliases['alias'].max()
        latest_rev = self.aliases[self.aliases['alias'] == latest_alias]['revision'].max()
        return self._version == latest_alias and self.revision == latest_rev

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_versions(cls, root_folder: Optional[str | Path] = None) -> pd.DataFrame:
        """Return a DataFrame of all known versions, annotated with download status.

        :param root_folder: directory that contains the Papyrus data tree
            (default: pystow's home directory)
        """
        dwnld_versions = get_downloaded_versions(root_folder)
        df = (
            cls.aliases
            .assign(version_long_fmt=cls.aliases['version'] + '.' + cls.aliases['revision'].astype(str))
            .rename(columns={'version': 'version_old_format', 'alias': 'version'})
            [['version_long_fmt', 'version', 'revision', 'version_old_format',
              'chembl', 'chembl_version', 'excape', 'sharma', 'christmann',
              'klaeger', 'merget', 'pickett']]
            .set_index('version_long_fmt')
        )
        return df.assign(
            downloaded=[PapyrusVersion(version=v) in dwnld_versions for v in df.index]
        )

    # ------------------------------------------------------------------
    # Instance method
    # ------------------------------------------------------------------

    def is_downloaded(self, root_folder: Optional[str | Path] = None) -> bool:
        """Return True if this version has been downloaded locally.

        :param root_folder: directory that contains the Papyrus data tree
            (default: pystow's home directory)
        """
        return self in get_downloaded_versions(root_folder)

    # ------------------------------------------------------------------
    # Sorting / comparison
    # ------------------------------------------------------------------

    def _sort_key(self) -> List[int]:
        return [int(u) for u in self.version.split('.')]

    def __lt__(self, other: PapyrusVersion) -> bool:
        return self._sort_key() < other._sort_key()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PapyrusVersion):
            return self.version == other.version and self.revision == other.revision
        if isinstance(other, str):
            return self == PapyrusVersion(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.version)

    def __repr__(self) -> str:
        return f'<PapyrusVersion version={self.version}>'


# ---------------------------------------------------------------------------
# Version-resolution helpers
# ---------------------------------------------------------------------------

def _sort_versions(versions: List[PapyrusVersion]) -> List[PapyrusVersion]:
    """Return *versions* sorted from oldest to newest."""
    return sorted(versions)


def get_online_versions() -> List[PapyrusVersion]:
    """Return all Papyrus versions available for download, oldest first."""
    links = get_papyrus_links()
    return _sort_versions([PapyrusVersion(version=v) for v in links.keys()])


def get_latest_online_version() -> PapyrusVersion:
    """Return the newest Papyrus version available for download."""
    return get_online_versions()[-1]


def _set_root_folder(root_folder: Optional[str | Path] = None):
    """Set the root folder for Papyrus data tree."""
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(
            root_folder if isinstance(root_folder, str) else root_folder.as_posix()
        )
    elif os.environ.get('PYSTOW_HOME') is not None:
        del os.environ['PYSTOW_HOME']


def get_downloaded_versions(root_folder: Optional[str | Path] = None) -> List[PapyrusVersion]:
    """Return all locally downloaded Papyrus versions, oldest first.

    :param root_folder: directory that contains the Papyrus data tree
        (default: pystow's home directory)
    """
    _set_root_folder(root_folder)
    version_json = pystow.join('papyrus', name='versions.json').as_posix()
    raw_versions: list = read_jsonfile(version_json)
    if not raw_versions:
        return []
    return _sort_versions([PapyrusVersion(version=v) for v in raw_versions])


def get_latest_downloaded_version(root_folder: Optional[str | Path] = None) -> PapyrusVersion:
    """Return the newest locally downloaded Papyrus version.

    :param root_folder: directory that contains the Papyrus data tree
        (default: pystow's home directory)
    :raises IOError: if no version has been downloaded yet
    """
    versions = get_downloaded_versions(root_folder)
    if not versions:
        raise IOError('No Papyrus data found locally (did you download it first?)')
    return versions[-1]


def is_local_version_available(
        version: str | PapyrusVersion,
        root_folder: Optional[str | Path] = None,
) -> bool:
    """Return True if *version* has been downloaded locally.

    :param version: version to check; accepts any form accepted by
        :class:`PapyrusVersion`
    :param root_folder: directory that contains the Papyrus data tree
        (default: pystow's home directory)
    """
    try:
        pv = version if isinstance(version, PapyrusVersion) else PapyrusVersion(version=version)
        return pv in get_downloaded_versions(root_folder)
    except (IOError, ValueError):
        return False


def process_data_version(
        version: str | PapyrusVersion,
        root_folder: Optional[str | Path] = None,
) -> PapyrusVersion:
    """Validate *version* against locally available data and return a
    :class:`PapyrusVersion`.

    The special string ``'latest'`` resolves to the newest downloaded version.
    This function is the canonical way to turn any user-supplied version
    argument into a validated :class:`PapyrusVersion`.

    :param version: version to validate; may be a string or a
        :class:`PapyrusVersion` already
    :param root_folder: directory that contains the Papyrus data tree
        (default: pystow's home directory)
    :raises IOError: if no Papyrus data has been downloaded at all
    :raises ValueError: if *version* is not among the downloaded versions
    """
    pv = version if isinstance(version, PapyrusVersion) else PapyrusVersion(version=version)
    downloaded = get_downloaded_versions(root_folder)
    if not downloaded:
        raise IOError('No Papyrus data found locally (did you download it first?)')
    if pv not in downloaded:
        available = ', '.join(v.version for v in downloaded)
        raise ValueError(
            f'Version {pv.version!r} is not available locally.\n'
            f'Either download it, or use an already downloaded version.\n'
            f'Downloaded versions: [{available}]'
        )
    if not pv.is_latest:
        aliases = PapyrusVersion.aliases
        latest_alias = aliases['alias'].max()
        latest_rev = aliases[aliases['alias'] == latest_alias]['revision'].max()
        warnings.warn(
            f"Papyrus {pv.version!r} is not the latest release "
            f"(latest: '{latest_alias}.{latest_rev}'). "
            f"Consider upgrading to access the most recent data.",
            FutureWarning,
            stacklevel=2,
        )
    return pv


def papyrus_version_module(pv: PapyrusVersion, root_folder: Optional[str | Path] = None) -> pystow.Module:
    """Return the pystow :class:`~pystow.Module` for *pv*'s on-disk folder.

    This is the single place in the codebase that translates a
    :class:`PapyrusVersion` into the ``_version_old_fmt`` string required by
    pystow for path construction.

    :param pv: resolved :class:`PapyrusVersion`
    :param root_folder: folder containing the bioactivity dataset
        (default: pystow's home folder)
    """
    _set_root_folder(root_folder)
    return pystow.module('papyrus', pv.version_old_fmt)


# ---------------------------------------------------------------------------
# Downloaded-file inventory
# ---------------------------------------------------------------------------

def get_downloaded_papyrus_files(root_folder: Optional[str] = None) -> pd.DataFrame:
    """Return a DataFrame describing which Papyrus files have been downloaded.

    Columns: ``version`` (canonical string), ``short_name``, ``file_name``,
    ``downloaded`` (bool).

    :param root_folder: folder containing the bioactivity dataset
        (default: pystow's home folder)
    """
    downloaded_versions = get_downloaded_versions(root_folder)
    # links.json is keyed by the old-format version strings ('05.4' etc.)
    files_map = get_papyrus_links(offline=True)

    _TRACKED = {
        'papyrus++', '2D_papyrus', '3D_papyrus',
        '2D_structures', '3D_structures',
        '2D_fingerprint', '3D_fingerprint',
        '2D_mordred', '3D_mordred',
        '2D_cddd', '2D_mold2',
        'proteins', 'proteins_unirep', 'proteins_prodec',
    }

    FileInfo = namedtuple('FileInfo', ('version_str', 'pv', 'short_name', 'file_name'))
    file_infos: List[FileInfo] = []
    for pv in downloaded_versions:
        # Bug-fix: original used PapyrusVersion as a dict key into files_map,
        # which is keyed by old-format strings → KeyError.
        version_files = files_map.get(pv.version_old_fmt, {})
        for ftype, fdata in version_files.items():
            if ftype not in _TRACKED:
                continue
            entries = fdata if isinstance(fdata, list) else [fdata]
            for entry in entries:
                file_infos.append(FileInfo(pv.version_old_fmt, pv, ftype, entry['name']))

    rows = []
    for fi in file_infos:
        pattern = os.path.join(
            papyrus_version_module(fi.pv).base.as_posix(), '**', fi.file_name,
        )
        rows.append({
            'version': fi.pv.version,
            'short_name': fi.short_name,
            'file_name': fi.file_name,
            'downloaded': len(glob.glob(pattern, recursive=True)) > 0,
        }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# File-location helper
# ---------------------------------------------------------------------------

def locate_file(dirpath: str, regex_pattern: str) -> list[str]:
    """Return all files in *dirpath* whose names match *regex_pattern*.

    :param dirpath: directory to search (non-recursive)
    :param regex_pattern: regular expression matched against bare filenames
    :raises NotADirectoryError: if *dirpath* does not exist
    :raises FileNotFoundError: if no matching file is found
    """
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f'Directory does not exist: {dirpath}')
    matches = [
        os.path.join(dirpath, fname)
        for fname in os.listdir(dirpath)
        if re.search(regex_pattern, fname) and not fname.endswith(':ZoneIdentifier')
    ]
    if not matches:
        raise FileNotFoundError(f'No file matching {regex_pattern!r} found in {dirpath}')
    return matches


# ---------------------------------------------------------------------------
# Row-count helper
# ---------------------------------------------------------------------------

def get_num_rows_in_file(
        filetype: str,
        is3D: bool,
        descriptor_name: Optional[str] = None,
        version: str | PapyrusVersion = 'latest',
        plusplus: bool = True,
        root_folder: Optional[str] = None,
) -> int:
    """Return the number of data rows in a Papyrus file.

    :param filetype: one of ``'bioactivities'``, ``'structures'``,
        ``'descriptors'``
    :param is3D: whether to consider the stereochemistry-aware (3D) variant
    :param descriptor_name: required when *filetype* is ``'descriptors'``; one
        of ``'cddd'``, ``'mold2'``, ``'mordred'``, ``'fingerprint'``
    :param version: Papyrus version to query
    :param plusplus: when *filetype* is ``'bioactivities'``, use the
        high-quality Papyrus++ subset
    :param root_folder: folder containing the bioactivity dataset
        (default: pystow's home folder)
    :raises ValueError: on invalid *filetype* or *descriptor_name*
    """
    if filetype not in ('bioactivities', 'structures', 'descriptors'):
        raise ValueError(
            "filetype must be one of ['bioactivities', 'structures', 'descriptors']"
        )
    if filetype == 'descriptors' and descriptor_name not in ('cddd', 'mold2', 'mordred', 'fingerprint'):
        raise ValueError(
            "descriptor_name must be one of ['cddd', 'mold2', 'mordred', 'fingerprint'] "
            "when filetype is 'descriptors'"
        )

    pv = version if isinstance(version, PapyrusVersion) else PapyrusVersion(version=version)
    _set_root_folder(root_folder)

    json_file = papyrus_version_module(pv).join(name='data_size.json').as_posix()
    sizes = read_jsonfile(json_file)

    if filetype == 'bioactivities':
        if plusplus:
            return sizes.get('papyrus_++', sizes.get('papyrus++'))
        return sizes['papyrus_3D'] if is3D else sizes['papyrus_2D']
    if filetype == 'structures':
        return sizes['structures_3D'] if is3D else sizes['structures_2D']
    # filetype == 'descriptors'
    return {
        'cddd': sizes['cddd'],
        'mold2': sizes['mold2'],
        'fingerprint': sizes['E3FP'] if is3D else sizes['ECFP6'],
        'mordred': sizes['mordred_3D'] if is3D else sizes['mordred_2D'],
    }[descriptor_name]


# ---------------------------------------------------------------------------
# Compression-conversion utilities
# ---------------------------------------------------------------------------

def convert_xz_to_gz(
        input_file: str,
        output_file: str,
        compression_level: Optional[int] = 9,
        progress: bool = False,
) -> None:
    """Transcode an LZMA-compressed ``.xz`` file to a gzip-compressed file.

    :param input_file: path to the source ``.xz`` file
    :param output_file: path to write the ``.gz`` file
    :param compression_level: gzip compression level (1–9, default 9)
    :param progress: display a progress bar
    """
    if compression_level is None:
        compression_level = 9
    chunksize = 10 * 1_048_576  # 10 MB
    with (
        lzma.open(input_file, 'rb') as fh,
        gzip.open(output_file, 'wb', compresslevel=compression_level) as oh,
    ):
        if progress:
            pbar = tqdm(desc='Determining size', unit='B', unit_scale=True)
            size = fh.seek(0, 2)
            fh.seek(0, 0)
            pbar.set_description('Converting')
            pbar.total = size
        while True:
            chunk = fh.read(chunksize)
            if not chunk:
                if progress:
                    pbar.close()
                break
            written = oh.write(chunk)
            if progress:
                pbar.update(written)


def convert_gz_to_xz(
        input_file: str,
        output_file: str,
        compression_level: int = lzma.PRESET_DEFAULT,
        extreme: bool = False,
        progress: bool = False,
) -> None:
    """Transcode a gzip-compressed file to an LZMA-compressed ``.xz`` file.

    :param input_file: path to the source ``.gz`` file
    :param output_file: path to write the ``.xz`` file
    :param compression_level: LZMA compression preset (0–9, default 6)
    :param extreme: apply the LZMA extreme-compression flag
    :param progress: display a progress bar
    """
    if compression_level is None:
        compression_level = lzma.PRESET_DEFAULT
    preset = compression_level | lzma.PRESET_EXTREME if extreme else compression_level
    chunksize = 10 * 1_048_576  # 10 MB
    with (
        gzip.open(input_file, 'rb') as fh,
        lzma.open(output_file, 'wb', preset=preset) as oh,
    ):
        if progress:
            pbar = tqdm(desc='Determining size', unit='B', unit_scale=True)
            size = fh.seek(0, 2)
            fh.seek(0, 0)
            pbar.set_description('Converting')
            pbar.total = size
        while True:
            chunk = fh.read(chunksize)
            if not chunk:
                if progress:
                    pbar.close()
                break
            written = oh.write(chunk)
            if progress:
                pbar.update(written)
