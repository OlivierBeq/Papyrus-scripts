# -*- coding: utf-8 -*-

"""IO functions."""

from __future__ import annotations

import builtins
import gzip
import hashlib
import importlib
import inspect
import json
import lzma
import os
import re
import shutil
import uuid
import warnings
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pystow
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# HTTP session helpers
# ---------------------------------------------------------------------------

#: Identifiable default, per API etiquette for scientific APIs (RCSB PDB,
#: UniProt, ...): lets them contact us if a script misbehaves.
_DEFAULT_USER_AGENT = 'papyrus-scripts/{version} (+https://github.com/OlivierBeq/Papyrus-scripts)'

#: Static fallback for the randomised path, only used if fake_useragent's
#: bundled browser data can't be loaded (e.g. broken install).
_FALLBACK_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
)

#: Lazily instantiated so import never blocks on fake_useragent's own init,
#: and its cost is only paid if the randomised fallback actually triggers.
_user_agent_factory: Any | None = None

#: Status codes suggesting the request was rejected/throttled based on how
#: it looks, not a transient server error - worth retrying with a new UA.
_BLOCKED_STATUS_CODES = frozenset({403, 429})


def notebook_safe_ncols(ncols: int | None) -> int | None:
    """Return *ncols*, or ``None`` if ``tqdm.auto`` resolved to the notebook widget backend.

    tqdm.notebook treats ``ncols`` as a pixel width, not a character count,
    so a terminal-sized value squashes the widget. MRO check is needed since
    tqdm.auto's notebook class subclasses tqdm_notebook but reports
    ``__module__ == 'tqdm.auto'``.
    """
    from tqdm.notebook import tqdm_notebook
    if issubclass(tqdm, tqdm_notebook):
        return None
    return ncols


def widen_indeterminate_notebook_bar(pbar) -> None:
    """Undo tqdm.notebook's 20px widget width for a bar with no known total yet."""
    from tqdm.notebook import tqdm_notebook
    if not isinstance(pbar, tqdm_notebook) or pbar.total:
        return
    container = getattr(pbar, 'container', None)
    if container is not None and len(container.children) > 1:
        container.children[1].layout.width = ''


def get_user_agent(randomize: bool = False) -> str:
    """Return a User-Agent string to send with requests to external APIs.

    :param randomize: if ``True``, return a randomised, up-to-date browser
        User-Agent (via ``fake_useragent``) instead of the default
        identifiable ``papyrus-scripts/...`` string. Falls back to a static
        modern-browser string if ``fake_useragent`` cannot supply one (e.g.
        corrupted/missing bundled browser data).
    """
    if not randomize:
        from .. import __version__
        return _DEFAULT_USER_AGENT.format(version=__version__)
    global _user_agent_factory
    try:
        if _user_agent_factory is None:
            from fake_useragent import UserAgent
            _user_agent_factory = UserAgent()
        return _user_agent_factory.random
    except Exception:
        return _FALLBACK_USER_AGENT


def _make_ua_fallback_hook(session: requests.Session) -> Callable[..., requests.Response]:
    """Build a response hook that retries a blocked (403/429) request once with a randomised User-Agent.

    If the retry succeeds, *session*'s default header is updated so later
    requests on the same session skip straight to the working UA.
    """
    def hook(response: requests.Response, **kwargs) -> requests.Response:
        request = response.request
        if response.status_code not in _BLOCKED_STATUS_CODES or request.headers.get('X-Papyrus-UA-Retried'):
            return response
        retry_request = request.copy()
        retry_request.headers['User-Agent'] = get_user_agent(randomize=True)
        retry_request.headers['X-Papyrus-UA-Retried'] = '1'
        retried = session.send(retry_request, **kwargs)
        if retried.ok:
            session.headers['User-Agent'] = retry_request.headers['User-Agent']
        retried.history.append(response)
        return retried
    return hook


def new_session(retries: int = 5, backoff_factor: float = 0.25,
                 status_forcelist: tuple[int, ...] = (500, 502, 503, 504)) -> requests.Session:
    """Return a :class:`requests.Session` with retries and an identifiable User-Agent.

    Reusing one session across repeated requests to the same host keeps the
    underlying TCP connection(s) alive instead of reconnecting (and
    re-handshaking TLS) on every call.

    Starts with an identifiable ``papyrus-scripts/...`` User-Agent; on a
    blocked (403/429) response, retries once with a randomised browser-like
    UA and keeps using it for the rest of the session if that unblocks things.

    :param retries: total number of retries for failed requests
    :param backoff_factor: backoff factor applied between retry attempts
    :param status_forcelist: HTTP status codes that trigger a retry
    """
    session = requests.Session()
    session.headers.update({'User-Agent': get_user_agent()})
    session.hooks['response'].append(_make_ua_fallback_hook(session))
    adapter = HTTPAdapter(max_retries=Retry(
        total=retries, backoff_factor=backoff_factor, status_forcelist=list(status_forcelist),
    ))
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------

def sha256sum(filename: str | Path, blocksize: int = 65536) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(filename, "rb") as fh:
        for block in iter(lambda: fh.read(blocksize), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_sha256sum(filename: str | Path, sha256: str, blocksize: int = 65536) -> bool:
    """Return True if the file's SHA-256 matches *sha256*, False otherwise.

    :param filename: path to the file to check
    :param sha256: expected 64-character hex digest
    :param blocksize: read block size in bytes
    :raises ValueError: if *sha256* is not a 64-character string
    """
    if not (isinstance(sha256, str) and len(sha256) == 64):
        raise ValueError(f"SHA256 must be a 64-character hex string, got: {sha256!r}")
    return sha256sum(filename, blocksize) == sha256


def write_jsonfile(data: object, json_outfile: str | Path) -> None:
    """Serialise *data* to *json_outfile* with readable indentation."""
    with open(json_outfile, 'w') as fh:
        json.dump(data, fh, indent=4)


def read_jsonfile(json_infile: str | Path) -> dict | list:
    """Deserialise *json_infile* and return the resulting object.

    Returns an empty dict if the file does not exist.
    """
    if not Path(json_infile).is_file():
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
                    'type': obj.__name__,
                },
            }
        return json.JSONEncoder.default(self, obj)


class TypeDecoder(json.JSONDecoder):
    """Custom JSON decoder that deserialises Python type objects from values."""

    def __init__(self, *args, **kwargs):
        """Set object_hook to :meth:`object_hook` so types are deserialised on load."""
        super().__init__(*args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, obj):
        """Handle types."""
        if '__type__' not in obj:
            return obj
        module = obj['__type__']['module']
        type_ = obj['__type__']['type']
        if module == 'builtins':
            # NB: the bare `__builtins__` global is a dict (not the `builtins`
            # module) in every module except __main__, so it must not be used
            # here - it would raise AttributeError for any builtin type.
            return getattr(builtins, type_)
        return getattr(importlib.import_module(module), type_)


# ---------------------------------------------------------------------------
# Dtype-conversion helpers
# ---------------------------------------------------------------------------

_BUILTIN_TO_POLARS: dict = {
    str:   pl.Utf8,
    float: pl.Float64,
    int:   pl.Int64,
    bool:  pl.Boolean,
}

_NUMPY_TO_POLARS: dict = {
    np.float32: pl.Float32,
    np.float64: pl.Float64,
    np.int8:    pl.Int8,
    np.int16:   pl.Int16,
    np.int32:   pl.Int32,
    np.int64:   pl.Int64,
    np.uint8:   pl.UInt8,
    np.uint16:  pl.UInt16,
    np.uint32:  pl.UInt32,
    np.uint64:  pl.UInt64,
    np.bool_:   pl.Boolean,
    np.str_:    pl.Utf8,
    np.object_: pl.Utf8,
}

#: The Papyrus data_types.json files actually shipped with every release
#: store dtypes as plain lowercase type-name strings (e.g. "float", "int"),
#: never as the TypeEncoder/__type__ hydrated form TypeDecoder expects - so
#: without this table every value fell through to the `return pl.Utf8`
#: default below regardless of its real dtype, silently forcing every
#: schema-driven column (all Papyrus++ columns, every descriptor column) to
#: String/Utf8 on read.
_TYPE_NAME_TO_POLARS: dict = {
    'str':    pl.Utf8,
    'object': pl.Utf8,
    'float':  pl.Float64,
    'int':    pl.Int64,
    'bool':   pl.Boolean,
}


def to_polars_dtype(t: Any) -> type[pl.DataType]:
    """Convert a Python builtin/NumPy type, or its lowercase name as a string, to the nearest Polars dtype."""
    if isinstance(t, str):
        return _TYPE_NAME_TO_POLARS.get(t, pl.Utf8)
    if t in _BUILTIN_TO_POLARS:
        return _BUILTIN_TO_POLARS[t]
    for np_type, pl_type in _NUMPY_TO_POLARS.items():
        if t is np_type or (isinstance(t, type) and issubclass(t, np_type)):
            return pl_type
    return pl.Utf8


def to_polars_schema(dtypes: dict) -> dict:
    """Convert a ``{col: python_type}`` map to a ``{col: polars_dtype}`` schema."""
    return {col: to_polars_dtype(t) for col, t in dtypes.items()}


def load_data_type_schemas(source_module: pystow.Module) -> dict:
    """Read a version folder's ``data_types.json`` and return ``{section: {col: polars_dtype}}``.

    :param source_module: pystow module for the specific Papyrus version
    """
    dtype_file = source_module.join(name='data_types.json')
    with open(dtype_file) as fh:
        raw = json.load(fh, cls=TypeDecoder)
    return {
        key: to_polars_schema(val) if isinstance(val, dict) else val
        for key, val in raw.items()
    }


# ---------------------------------------------------------------------------
# Disk-space helpers
# ---------------------------------------------------------------------------

def enough_disk_space(destination: str | Path, required: int, margin: float = 0.10) -> bool:
    """Check whether *destination*'s drive has enough headroom.

    :param destination: folder (or file path) whose drive is checked
    :param required: bytes that will be written
    :param margin: fraction of total capacity to keep free after writing
    """
    total, _, free = shutil.disk_usage(destination)
    return free - required > margin * total


def get_disk_space(destination: str | Path) -> int:
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
    local_file = Path(__file__).parent / 'links.json'
    if not offline:
        url = "https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/db-links/links.json"
        try:
            response = new_session().get(url, verify=True)
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
    local_file = Path(__file__).parent / 'aliases.json'
    if not offline:
        url = "https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/db-links/aliases.json"
        try:
            response = new_session().get(url, verify=True)
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

    #: Attributes attached dynamically (via ``setattr``) during ``__init__``
    #: from the matched row of ``aliases`` - declared here purely so static
    #: type checkers know they exist.
    _version: str
    _version_old_fmt: str
    revision: str

    def __init__(
            self,
            version: str | None = None,
            revision: str | None = None,
            chembl_version: int | None = None,
            chembl: bool | None = None,
            excape: bool | None = None,
            sharma: bool | None = None,
            christmann: bool | None = None,
            klaeger: bool | None = None,
            merget: bool | None = None,
            pickett: bool | None = None,
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
                # revision is a string column; compare numerically so e.g. '10' > '9'
                latest_rev = max(
                    self.aliases[self.aliases['alias'] == latest_alias]['revision'], key=int,
                )
                query = f'alias == "{latest_alias}" and revision == "{latest_rev}"'
            elif version.count('.') == 2:
                if revision is not None:
                    raise ValueError(
                        'Revision number provided too many times '
                        '(as `revision` and as part of `version`)',
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
                matches = self.aliases[
                    (self.aliases['version'] == version) | (self.aliases['alias'] == version)
                ]
                # Only warn when there is real ambiguity to resolve - a
                # version/alias with a single known revision (true for every
                # old-format string, e.g. '05.7', and for every alias so far)
                # isn't "latest revision selected", it's the only one there is.
                if matches['revision'].nunique() > 1:
                    warnings.warn('Revision number not provided; latest revision selected.', stacklevel=2)
                latest_rev = max(matches['revision'], key=int)
                query = (
                    f'(version == "{version}" or alias == "{version}") '
                    f'and (revision == "{latest_rev}")'
                )
        else:
            predicates: list[str] = []
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
            # For 3-part canonical strings (e.g. '2022.04.1'), aliases.json may only
            # carry the latest revision.  Fall back to the alias row and override the
            # revision so older revisions can still be constructed.
            _fallback_handled = False
            if version is not None and version.count('.') == 2:
                parts = version.split('.')
                split_alias = '.'.join(parts[:2])
                split_revision = parts[2].lstrip('0')
                alias_rows = self.aliases[
                    (self.aliases['version'] == split_alias) | (self.aliases['alias'] == split_alias)
                ]
                if not alias_rows.empty:
                    row = alias_rows.iloc[0].copy()
                    if int(row['revision']) < int(split_revision):
                        raise ValueError(f'Revision {split_revision} is not available for version {row["alias"]}. '
                                         f'Latest revision: {row["revision"]}.')
                    row['revision'] = split_revision
                    subset = pd.DataFrame([row])
                    _fallback_handled = True
            if not _fallback_handled:
                raise ValueError('No Papyrus version matches the supplied information.')
        if len(subset) > 1:
            raise ValueError(
                'The supplied information matches multiple versions:\n\n'
                + str(
                    subset.drop(columns='version')
                    .rename(columns={'alias': 'version'})
                    .set_index('version'),
                )
                + '\n\nNarrow your criteria to select a single version.',
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

        # True only when the caller passed an old-format string (e.g. '05.4').
        # Used by pystow_path_key to preserve backward-compatible folder names.
        self._input_is_old_fmt: bool = (
            version is not None
            and version in self.aliases['version'].values
        )

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
    def pystow_path_key(self) -> str:
        """Key for the pystow directory that stores this version's files.

        Returns the old-format string (e.g. ``'05.4'``) when the version was
        originally specified in old format — preserving the on-disk layout
        created by the previous PyPI release.  Otherwise returns the canonical
        new-format string (e.g. ``'2022.04.2'``), which always includes the
        revision number so that different revisions of the same alias never
        share a folder.
        """
        return self._version_old_fmt if self._input_is_old_fmt else self.version

    @property
    def is_latest(self) -> bool:
        """Return True if this is the most recent version in the known releases table."""
        latest_alias = self.aliases['alias'].max()
        latest_rev = max(self.aliases[self.aliases['alias'] == latest_alias]['revision'], key=int)
        return self._version == latest_alias and self.revision == latest_rev

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def get_versions(cls, root_folder: str | Path | None = None) -> pd.DataFrame:
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
            downloaded=[PapyrusVersion(version=v) in dwnld_versions for v in df.index],
        )

    # ------------------------------------------------------------------
    # Instance method
    # ------------------------------------------------------------------

    def is_downloaded(self, root_folder: str | Path | None = None) -> bool:
        """Return True if this version has been downloaded locally.

        :param root_folder: directory that contains the Papyrus data tree
            (default: pystow's home directory)
        """
        return self in get_downloaded_versions(root_folder)

    # ------------------------------------------------------------------
    # Sorting / comparison
    # ------------------------------------------------------------------

    def _sort_key(self) -> list[int]:
        return [int(u) for u in self.version.split('.')]

    def __lt__(self, other: PapyrusVersion) -> bool:
        """Compare numerically component-by-component (not lexicographically)."""
        return self._sort_key() < other._sort_key()

    def __eq__(self, other: object) -> bool:
        """Compare by version+revision; a plain string is parsed first."""
        if isinstance(other, PapyrusVersion):
            return self.version == other.version and self.revision == other.revision
        if isinstance(other, str):
            return self == PapyrusVersion(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Hash by version only (coarser than __eq__, which also checks revision)."""
        return hash(self.version)

    def __str__(self) -> str:
        """Return the version string."""
        return self.version

    def __repr__(self) -> str:
        """Return an unambiguous, eval-able-looking representation."""
        return f'<PapyrusVersion version={self.version}>'


# ---------------------------------------------------------------------------
# Version-resolution helpers
# ---------------------------------------------------------------------------

def _sort_versions(versions: list[PapyrusVersion]) -> list[PapyrusVersion]:
    """Return *versions* sorted from oldest to newest."""
    return sorted(versions)


def get_online_versions() -> list[PapyrusVersion]:
    """Return all Papyrus versions available for download, oldest first."""
    links = get_papyrus_links()
    return _sort_versions([PapyrusVersion(version=v) for v in links.keys()])


def get_latest_online_version() -> PapyrusVersion:
    """Return the newest Papyrus version available for download."""
    return get_online_versions()[-1]


def _set_root_folder(root_folder: str | Path | None = None):
    """Set the root folder for Papyrus data tree."""
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = str(Path(root_folder).resolve())
    elif os.environ.get('PYSTOW_HOME') is not None:
        del os.environ['PYSTOW_HOME']


def get_downloaded_versions(root_folder: str | Path | None = None) -> list[PapyrusVersion]:
    """Return all locally downloaded Papyrus versions, oldest first.

    :param root_folder: directory that contains the Papyrus data tree
        (default: pystow's home directory)
    """
    _set_root_folder(root_folder)
    version_json = pystow.join('papyrus', name='versions.json')
    # versions.json is always written as a JSON array (see _update_versions_json).
    raw_versions = cast(list, read_jsonfile(version_json))
    if not raw_versions:
        return []
    return _sort_versions([PapyrusVersion(version=v) for v in raw_versions])


def get_latest_downloaded_version(root_folder: str | Path | None = None) -> PapyrusVersion:
    """Return the newest locally downloaded Papyrus version.

    :param root_folder: directory that contains the Papyrus data tree
        (default: pystow's home directory)
    :raises IOError: if no version has been downloaded yet
    """
    versions = get_downloaded_versions(root_folder)
    if not versions:
        raise OSError('No Papyrus data found locally (did you download it first?)')
    return versions[-1]


def is_local_version_available(
        version: str | PapyrusVersion,
        root_folder: str | Path | None = None,
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
    except (OSError, ValueError):
        return False


def process_data_version(
        version: str | PapyrusVersion,
        root_folder: str | Path | None = None,
) -> PapyrusVersion:
    """Validate *version* against locally available data and return a :class:`PapyrusVersion`.

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
    downloaded = get_downloaded_versions(root_folder)
    if not downloaded:
        raise OSError('No Papyrus data found locally (did you download it first?)')
    if isinstance(version, str) and version == 'latest':
        # get_downloaded_versions() returns oldest-first: newest is last.
        pv = downloaded[-1]
    else:
        pv = version if isinstance(version, PapyrusVersion) else PapyrusVersion(version=version)
    if pv not in downloaded:
        available = ', '.join(v.version for v in downloaded)
        raise ValueError(
            f'Version {pv.version!r} is not available locally.\n'
            f'Either download it, or use an already downloaded version.\n'
            f'Downloaded versions: [{available}]',
        )
    if not pv.is_latest:
        aliases = PapyrusVersion.aliases
        latest_alias = aliases['alias'].max()
        latest_rev = max(aliases[aliases['alias'] == latest_alias]['revision'], key=int)
        warnings.warn(
            f"Papyrus {pv.version!r} is not the latest release "
            f"(latest: '{latest_alias}.{latest_rev}'). "
            f"Consider upgrading to access the most recent data.",
            FutureWarning,
            stacklevel=2,
        )
    return pv


def papyrus_version_module(pv: PapyrusVersion, root_folder: str | Path | None = None) -> pystow.Module:
    """Return the pystow :class:`~pystow.Module` for *pv*'s on-disk folder.

    This is the single place in the codebase that translates a
    :class:`PapyrusVersion` into the ``pystow_path_key`` string required by
    pystow for path construction - matching the folder name ``download_papyrus``
    actually writes to (old-format only when *pv* was itself specified in old
    format; the canonical new-format string otherwise).

    :param pv: resolved :class:`PapyrusVersion`
    :param root_folder: folder containing the bioactivity dataset
        (default: pystow's home folder)
    """
    _set_root_folder(root_folder)
    return pystow.module('papyrus', pv.pystow_path_key)


# ---------------------------------------------------------------------------
# Downloaded-file inventory
# ---------------------------------------------------------------------------

def get_downloaded_papyrus_files(root_folder: str | Path | None = None) -> pd.DataFrame:
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
    file_infos: list[FileInfo] = []
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
        matches = papyrus_version_module(fi.pv, root_folder=root_folder).base.glob(f'**/{fi.file_name}')
        rows.append({
            'version': fi.pv.version,
            'short_name': fi.short_name,
            'file_name': fi.file_name,
            'downloaded': next(matches, None) is not None,
        },
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# File-location helper
# ---------------------------------------------------------------------------

def locate_file(dirpath: str | Path, regex_pattern: str) -> list[Path]:
    """Return all files in *dirpath* whose names match *regex_pattern*.

    :param dirpath: directory to search (non-recursive)
    :param regex_pattern: regular expression matched against bare filenames
    :raises NotADirectoryError: if *dirpath* does not exist
    :raises FileNotFoundError: if no matching file is found
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise NotADirectoryError(f'Directory does not exist: {dirpath}')
    matches = [
        entry
        for entry in dirpath.iterdir()
        if re.search(regex_pattern, entry.name) and not entry.name.endswith(':ZoneIdentifier')
    ]
    if not matches:
        raise FileNotFoundError(f'No file matching {regex_pattern!r} found in {dirpath}')
    return matches


def _prefer_parquet(files: list[Path]) -> Path:
    """Return the ``.parquet`` file among *files* if one is present, else the first match."""
    for f in files:
        if f.suffix == '.parquet':
            return f
    return files[0]


# ---------------------------------------------------------------------------
# Row-count helper
# ---------------------------------------------------------------------------

def get_num_rows_in_file(
        filetype: str,
        is3D: bool,
        descriptor_name: str | None = None,
        version: str | PapyrusVersion = 'latest',
        plusplus: bool = True,
        root_folder: str | Path | None = None,
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
            "filetype must be one of ['bioactivities', 'structures', 'descriptors']",
        )
    if filetype == 'descriptors' and descriptor_name not in ('cddd', 'mold2', 'mordred', 'fingerprint'):
        raise ValueError(
            "descriptor_name must be one of ['cddd', 'mold2', 'mordred', 'fingerprint'] "
            "when filetype is 'descriptors'",
        )

    pv = version if isinstance(version, PapyrusVersion) else PapyrusVersion(version=version)

    json_file = papyrus_version_module(pv, root_folder=root_folder).join(name='data_size.json')
    # data_size.json is always written as a JSON object.
    sizes = cast(dict, read_jsonfile(json_file))

    if filetype == 'bioactivities':
        if plusplus:
            if 'papyrus_++' in sizes:
                return sizes['papyrus_++']
            if 'papyrus++' in sizes:
                return sizes['papyrus++']
            raise KeyError(
                "Neither 'papyrus_++' nor 'papyrus++' found in data_size.json",
            )
        return sizes['papyrus_3D'] if is3D else sizes['papyrus_2D']
    if filetype == 'structures':
        return sizes['structures_3D'] if is3D else sizes['structures_2D']
    # filetype == 'descriptors' - descriptor_name was validated above.
    if descriptor_name is None:
        raise RuntimeError("descriptor_name unexpectedly None despite filetype == 'descriptors' validation")
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
        input_file: str | Path,
        output_file: str | Path,
        compression_level: int | None = 9,
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
            widen_indeterminate_notebook_bar(pbar)
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


#: Maps a polars dtype (as used in schema_overrides) to the pandas dtype
#: name that preserves it through pd.read_csv -> pyarrow.Table.from_pandas.
#: Nullable pandas extension types (capitalised: 'Int64', 'boolean', ...)
#: are used for anything that isn't already nullable by nature (float,
#: string) - a plain numpy 'int64' column can't hold NaN, so a chunk with
#: even one missing value would silently upcast to float64 while a
#: different chunk without one stays int64, and ParquetWriter.write_table()
#: rejects the resulting schema mismatch between chunks.
_POLARS_TO_PANDAS_DTYPE: dict = {
    pl.Utf8: 'string',
    pl.Float32: 'float32',
    pl.Float64: 'float64',
    pl.Int8: 'Int8',
    pl.Int16: 'Int16',
    pl.Int32: 'Int32',
    pl.Int64: 'Int64',
    pl.UInt8: 'UInt8',
    pl.UInt16: 'UInt16',
    pl.UInt32: 'UInt32',
    pl.UInt64: 'UInt64',
    pl.Boolean: 'boolean',
}

#: The nullable-integer pandas dtype names from the mapping above. A column
#: whose real values include a decimal point, "NaN", or scientific notation
#: (schema mistakes happen - the "right" dtype for a column can differ
#: between Papyrus releases) crashes pandas' C parser outright if handed one
#: of these dtypes directly ("cannot safely cast non-equivalent float64 to
#: int64"), so convert_xz_to_parquet never does: see its docstring.
_NULLABLE_INT_DTYPES: frozenset = frozenset({
    'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64',
})


def _schema_overrides_to_pandas_dtype(schema_overrides: dict | None) -> dict | None:
    """Convert a ``{column: polars_dtype}`` mapping to pandas dtype names."""
    if not schema_overrides:
        return None
    return {
        col: _POLARS_TO_PANDAS_DTYPE.get(dtype, 'string')
        for col, dtype in schema_overrides.items()
    }


def _coerce_object_columns_to_string(chunk: pd.DataFrame) -> None:
    """Cast every ``object``-dtyped column of *chunk* to pandas' nullable ``'string'`` dtype, in place.

    A column pandas couldn't type uniformly (e.g. some rows numeric-looking,
    others not) comes back ``object``, mixing real Python ``int``/``str``
    values - ``pyarrow.Table.from_pandas`` fails outright on that mix.
    ``'string'`` stringifies every value the same way and maps NaN-likes to
    ``pd.NA`` rather than the literal text ``"nan"``.
    """
    for col in chunk.columns[chunk.dtypes == object]:  # noqa: E721 - vectorized pandas dtype comparison, not a type identity check
        chunk[col] = chunk[col].astype('string')


def _downcast_integer_overrides(
        chunk: pd.DataFrame,
        overrides: dict,
        forced_float_cols: set,
) -> str | None:
    """Downcast *chunk*'s int-schema columns to their real nullable integer dtype, in place.

    Every column in *overrides* mapped to one of :data:`_NULLABLE_INT_DTYPES`
    is read leniently as ``'Float64'`` (see :func:`convert_xz_to_parquet`),
    so it must be cast back to its intended integer dtype here once the
    chunk is safely in hand. A column already in *forced_float_cols* is
    known to hold genuine fractional values from an earlier chunk and is
    left as-is.

    :returns: the name of the first column found to hold a real fractional
        value (not yet in *forced_float_cols*, so this is news) - the
        caller must restart the conversion with it added there - or
        ``None`` if every eligible column downcast cleanly.
    """
    for col, target in overrides.items():
        if target not in _NULLABLE_INT_DTYPES or col in forced_float_cols:
            continue
        non_null = chunk[col].dropna()
        if len(non_null) == 0 or (non_null % 1 == 0).all():
            chunk[col] = chunk[col].astype(target)
        else:
            return col
    return None


def _first_type_mismatch(established: pa.Schema, candidate: pa.Schema) -> str | None:
    """Return the name of the first field whose type in *candidate* differs from *established*, else None."""
    for field in candidate:
        if established.field(field.name).type != field.type:
            return field.name
    return None


def convert_xz_to_parquet(
        input_file: str | Path,
        output_file: str | Path,
        separator: str = '\t',
        schema_overrides: dict | None = None,
        null_values: list | str | None = None,
        progress: bool = False,
        chunksize: int = 250_000,
        desc: str | None = None,
        position: int = 0,
        total: int | None = None,
        leave: bool = True,
        ncols: int = 100,
        on_progress: Callable[[int], None] | None = None,
        on_reset: Callable[[], None] | None = None,
) -> None:
    r"""Losslessly convert an LZMA-compressed CSV/TSV file to Parquet.

    Reads *input_file* in row-bounded chunks via ``pandas.read_csv``
    (which decompresses ``.xz`` on the fly - no intermediate decompressed
    file ever touches disk) and writes each chunk immediately via a single
    streaming ``pyarrow.parquet.ParquetWriter``, so memory and extra disk
    usage stay close to *chunksize* regardless of the true file size.

    A prior implementation decompressed to a temporary file first and used
    Polars' ``scan_csv(...).sink_parquet(...)`` (splitting into row-bounded
    pieces above 2 GiB when that alone still wasn't enough) - benchmarked
    against this one on an 8M-row/3.3 GB-decompressed synthetic file:
    peak RSS 3.2 GB vs. 379 MB here, peak extra disk 6.7 GB vs. 50 MB here,
    for +3.7s of wall time here. The old approach also OOM-killed a real
    39-47 GB Papyrus 3D bioactivity file (version 2024.09.1) outright on a
    31 GB-RAM machine; this one has no such failure mode since it never
    materialises more than *chunksize* rows at once.

    :param input_file: path to the source ``.xz`` file
    :param output_file: path to write the ``.parquet`` file
    :param separator: field separator of the decompressed CSV/TSV content
    :param schema_overrides: ``{column: polars_dtype}`` overrides, converted
        to pandas dtypes and forwarded to ``pd.read_csv``; a column not
        covered here has its dtype inferred by pandas, independently for
        each chunk. Two failure modes follow from that and are both handled
        automatically: a column pandas couldn't type uniformly *within* one
        chunk (e.g. a "years" column holding a bare ``"1990"`` in some rows
        and ``"1990;1995"`` in others, mixing real Python ``int`` and
        ``str`` values) comes back ``object``-dtyped and is coerced to
        pandas' nullable ``'string'`` dtype before handing the chunk to
        ``pyarrow``, which otherwise rejects that mix outright; a column
        inferred with two *different* uniform dtypes *across* chunks (e.g.
        ``int64`` in one, ``float64`` in another because only that chunk has
        a null, or ``string`` in one and ``int64`` in another because only
        that chunk mixes non-numeric values in) is detected once the
        mismatch would otherwise break ``ParquetWriter``, and the whole
        conversion is restarted with that column forced to ``'string'``
        from the first row - the safe common type for any two dtypes here.
        A column *covered* here as one of the nullable integer dtypes
        (``pl.Int8``/``pl.UInt64``/etc.) is never handed to ``pd.read_csv``
        as such directly - it's read as ``'Float64'`` and
        cast back to the intended integer dtype once each chunk is safely
        parsed. Real Papyrus releases have shipped a column documented as
        integer in ``data_types.json`` that actually holds float-formatted
        values (e.g. ``"1.0"``) for a given release; handing pandas the
        strict integer dtype directly crashes its C parser outright
        (``TypeError: cannot safely cast non-equivalent float64 to
        int64``) the moment it hits such a value, before any of the above
        chunk-level handling ever gets a chance to run. A column found to
        hold a genuine fraction is left ``'Float64'`` from the first row
        onward instead (same restart-with-a-wider-type approach as above).
    :param null_values: values to treat as null, forwarded to ``pd.read_csv``
        as ``na_values`` (with ``keep_default_na=False``, so nothing beyond
        what's listed here is ever treated as null). Defaults to
        ``['', 'NA']`` when omitted (``None``) - real Papyrus releases have
        shipped columns absent from data_types.json holding literal "NA"
        for missing values (e.g. a "Year" column on the 3D bioactivity file
        of version 2024.09.1). Pass ``[]`` explicitly to keep empty strings
        and literal "NA" values as-is instead of null.
    :param progress: display a progress bar while converting
    :param chunksize: rows read and written per chunk
    :param desc: progress bar label; defaults to *input_file*'s name so
        converting many files in a row (e.g. from :func:`download_papyrus`)
        shows which one is currently in progress rather than a bare
        "Converting" repeated identically for every file
    :param position: ``tqdm`` line offset - pass a nonzero value when this
        bar must coexist on-screen with another (already-running) one, e.g.
        a download's overall byte-progress bar, so they render on separate
        lines instead of overwriting each other
    :param total: expected row count, shown as "N/total" plus a percentage
        instead of a bare running count; an approximation (e.g. from a
        released version's ``data_size.json``) is fine - it only affects
        the display, never the actual conversion. ``None`` (default) falls
        back to an open-ended count when the true size isn't known upfront.
    :param leave: keep the finished bar visible on-screen after this
        function returns; set to False when converting many files back to
        back (e.g. from :func:`download_papyrus`) so completed per-file
        bars don't pile up as one leftover line each
    :param ncols: fixed bar width in characters, so it never grows to fill
        the whole terminal on a wide screen
    :param on_progress: called with the row count of each chunk just
        written (a delta, not a running total) - lets a caller running this
        in its own process (e.g. download_papyrus's converter subprocess)
        report progress back to a single process that owns every tqdm bar,
        instead of rendering its own bar here. Two independent processes
        each rendering position-pinned tqdm bars (this function's own
        ``progress=True`` path, used concurrently with another bar in a
        different process) drift out of sync over time - each one's
        cursor-movement math assumes it is the only writer, so once one
        process's bar has scrolled the terminal in a way the other doesn't
        know about, stray blank lines accumulate. Used together with
        *progress*\ =False so no bar is created here at all.
    :param on_reset: called (with no arguments) whenever a conversion
        attempt restarts from row 0 because of dtype drift - the
        conversion-progress counterpart of *on_progress*, so the receiving
        side can reset its own bar back to 0 in step.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    # Written under a different name and only moved to *output_file* (an
    # atomic filesystem rename) once every chunk has been written - a plain
    # try/finally cleanup is not enough here, since a hard kill (e.g. the
    # OOM killer) skips all Python exception handling entirely and would
    # still leave a truncated, invalid file sitting at *output_file* for
    # download_papyrus's "does the .parquet already exist" skip-check to
    # mistake for a completed conversion on every subsequent run.
    # Unique per call (pid + random suffix) so two concurrent conversions of
    # the same output_file (e.g. two processes lazily reading it via
    # reader.py) never collide; any other stale '.converting' file from a
    # prior crashed run is cleaned up below too, so it can't accumulate.
    tmp_parquet = output_file.with_name(f'{output_file.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.converting')
    for stale in output_file.parent.glob(f'{output_file.name}.*.converting'):
        stale.unlink(missing_ok=True)
    na_values = null_values if null_values is not None else ['', 'NA']
    overrides = _schema_overrides_to_pandas_dtype(schema_overrides) or {}
    forced_string_cols: set[str] = set()
    forced_float_cols: set[str] = set()
    if desc is None:
        desc = f'Converting {input_file.name}'

    pbar = tqdm(
        desc=desc, unit=' rows', unit_scale=True, position=position,
        total=total, leave=leave, ncols=notebook_safe_ncols(ncols),
    ) if progress else None
    if pbar is not None:
        widen_indeterminate_notebook_bar(pbar)
    try:
        while True:
            # Every int-schema column is read as 'Float64' - never its real
            # nullable integer dtype - regardless of forced_float_cols: see
            # the docstring and _downcast_integer_overrides, which casts it
            # back after each chunk is safely in hand for every column not
            # already confirmed (by a past attempt) to hold real fractions.
            dtype = {
                col: ('Float64' if target in _NULLABLE_INT_DTYPES else target)
                for col, target in overrides.items()
            }
            dtype.update(dict.fromkeys(forced_string_cols, 'string'))
            writer = None
            first_schema: pa.Schema | None = None
            drift: tuple[str, str] | None = None
            tmp_parquet.unlink(missing_ok=True)
            if pbar is not None:
                pbar.reset()
            if on_reset is not None:
                on_reset()
            try:
                # Default quoting (respecting '"' as the CSV quote character) is
                # deliberately left enabled: some Papyrus columns (InChI_AuxInfo,
                # doc_id/citation fields) legitimately hold values that embed a
                # literal '"' and even a literal newline, properly RFC4180-quoted
                # by the exporter - disabling quoting (tried and reverted) does
                # NOT just change a row count, it corrupts those specific records
                # by splitting one logical row into several garbage fragments
                # (confirmed on 05.5++_combined_set_without_stereochemistry.tsv
                # of 2022.08.1: record BXNJHAXVSOCGBA_on_P68400_WT, cleanly
                # reconstructed with quoting enabled, became 3 rows of mostly
                # NaN/stray-quote garbage with quoting disabled). The resulting
                # row count is lower than data_size.json's published count for
                # this same reason - data_size.json's numbers are a naive
                # physical line count, not quote-aware logical records.
                # low_memory=False: pandas' C parser otherwise processes each
                # chunksize-bounded chunk in smaller internal blocks of its
                # own and infers dtypes per block, emitting a DtypeWarning
                # straight to stderr (bypassing tqdm's write hooks) whenever
                # two blocks of the *same* chunk disagree - on top of being
                # spurious noise, that raw write lands mid-redraw of the
                # progress bar above and forces tqdm to reprint a duplicate
                # line. chunksize already bounds how much a single block
                # reads at once, so disabling the extra internal splitting
                # costs no meaningful memory here.
                reader = pd.read_csv(
                    input_file, sep=separator, compression='xz', chunksize=chunksize,
                    dtype=dtype, na_values=na_values, keep_default_na=False,
                    low_memory=False,
                )
                for chunk in reader:
                    _coerce_object_columns_to_string(chunk)
                    float_drift_col = _downcast_integer_overrides(chunk, overrides, forced_float_cols)
                    if float_drift_col is not None:
                        drift = (float_drift_col, 'float')
                        break
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_parquet, table.schema)
                        first_schema = table.schema
                    else:
                        mismatch_col = _first_type_mismatch(first_schema, table.schema)
                        if mismatch_col is not None:
                            drift = (mismatch_col, 'string')
                            break
                    writer.write_table(table)
                    if pbar is not None:
                        pbar.update(len(chunk))
                    if on_progress is not None:
                        on_progress(len(chunk))
                if drift is None and writer is None:  # pragma: no cover
                    # Defensive: a header-only input still yields one empty
                    # chunk from pd.read_csv(chunksize=...) (confirmed - so
                    # writer is always set above), never zero chunks, but
                    # keep this as a safety net in case that ever changes.
                    empty = pd.read_csv(
                        input_file, sep=separator, compression='xz', nrows=0, dtype=dtype,
                    )
                    _coerce_object_columns_to_string(empty)
                    pq.write_table(pa.Table.from_pandas(empty, preserve_index=False), tmp_parquet)
            finally:
                if writer is not None:
                    writer.close()
            if drift is None:
                break
            drifted_col, kind = drift
            (forced_string_cols if kind == 'string' else forced_float_cols).add(drifted_col)
        tmp_parquet.replace(output_file)
    finally:
        if pbar is not None:
            pbar.close()
        tmp_parquet.unlink(missing_ok=True)


#: Matches an SD tag line, e.g. ``> <InChIKey>`` or ``>  <Name> (1)``.
_SD_PROP_PATTERN = re.compile(r'^>.*?<([^>]+)>')

#: SD-record delimiter and the two accepted spellings of the CTAB terminator.
_SD_RECORD_END = '$$$$'
_SD_CTAB_END_MARKERS = ('M  END', 'M END')


def _split_ctab_and_properties(lines: list[str]) -> tuple[str, dict[str, str]]:
    """Split one SD record's lines into its CTAB block and SD tag properties.

    :param lines: lines of a single record, not including the trailing
        ``$$$$`` delimiter
    :returns: ``(ctab, properties)``, mapping each ``> <TAG>`` name to its
        (stripped) value
    """
    end_idx = next(
        (i for i, line in enumerate(lines) if line.startswith(_SD_CTAB_END_MARKERS)),
        None,
    )
    if end_idx is None:
        # No 'M  END': end the CTAB at the first '>' tag, or the whole block.
        end_idx = next((i - 1 for i, line in enumerate(lines) if line.startswith('>')), len(lines) - 1)

    ctab = ''.join(lines[:end_idx + 1])

    properties: dict[str, str] = {}
    current_prop: str | None = None
    current_value: list[str] = []
    for line in lines[end_idx + 1:]:
        if line.startswith('>'):
            if current_prop is not None:
                properties[current_prop] = ''.join(current_value).strip()
            match = _SD_PROP_PATTERN.match(line)
            current_prop = match.group(1) if match else None
            current_value = []
        elif current_prop is not None:
            current_value.append(line)
    if current_prop is not None:
        properties[current_prop] = ''.join(current_value).strip()
    return ctab, properties


def convert_sd_to_parquet(
        input_file: str | Path,
        output_file: str | Path,
        chunksize: int = 20_000,
        progress: bool = False,
        desc: str | None = None,
        position: int = 0,
        total: int | None = None,
        leave: bool = True,
        ncols: int = 100,
        on_progress: Callable[[int], None] | None = None,
) -> None:
    """Losslessly convert an LZMA-compressed SD file to a structures Parquet file.

    Each row holds one molecule's SD tag properties as string columns, plus
    a ``'ctab'`` column with the raw MDL connection-table block, stored as
    ``large_binary`` (avoids the ~2 GB per-array overflow risk of a plain
    32-bit-offset string/binary column at multi-million-row scale, and
    skips UTF-8 validation - a CTAB is opaque bytes, and RDKit's
    ``MolFromMolBlock`` accepts them directly). Parsing is a plain text
    split on SD's own ``$$$$``/``M  END``/``> <TAG>`` structure - no RDKit
    involved, so the CTAB is preserved exactly and no molecule is built.

    Two passes: the first scans for the union of SD tag names (needed to
    fix the Parquet schema upfront), the second streams and writes
    *chunksize*-row batches.

    :param input_file: path to the source ``.sd.xz`` file
    :param output_file: path to write the ``.parquet`` file (written
        atomically, like :func:`convert_xz_to_parquet`)
    :param chunksize: molecules per write batch
    :param progress: display a progress bar while converting
    :param desc: progress bar label; defaults to *input_file*'s name
    :param position: ``tqdm`` line offset, for coexisting with another bar
    :param total: expected molecule count, shown in the bar; an
        approximation is fine
    :param leave: keep the finished bar visible after this function returns
    :param ncols: fixed bar width in characters
    :param on_progress: called with each chunk's row count (a delta) - lets
        a caller in its own process report progress to the process
        rendering the bar, instead of rendering one here
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    tmp_parquet = output_file.with_name(f'{output_file.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.converting')
    for stale in output_file.parent.glob(f'{output_file.name}.*.converting'):
        stale.unlink(missing_ok=True)
    if desc is None:
        desc = f'Converting {input_file.name}'

    properties: list[str] = []
    seen: set[str] = set()
    with lzma.open(input_file, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                match = _SD_PROP_PATTERN.match(line)
                if match:
                    name = match.group(1)
                    if name not in seen:
                        seen.add(name)
                        properties.append(name)

    schema = pa.schema(
        [pa.field(name, pa.string()) for name in properties]
        + [pa.field('ctab', pa.large_binary())],
    )

    def _write_chunk(writer: pq.ParquetWriter, records: list[tuple[str, dict]]) -> None:
        columns: dict[str, list] = {name: [] for name in properties}
        ctabs: list[bytes] = []
        for ctab, props in records:
            for name in properties:
                columns[name].append(props.get(name))
            ctabs.append(ctab.encode('utf-8'))
        arrays = [pa.array(columns[name], type=pa.string()) for name in properties]
        arrays.append(pa.array(ctabs, type=pa.large_binary()))
        writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
        if pbar is not None:
            pbar.update(len(records))
        if on_progress is not None:
            on_progress(len(records))

    pbar = tqdm(
        desc=desc, unit=' mols', unit_scale=True, position=position,
        total=total, leave=leave, ncols=notebook_safe_ncols(ncols),
    ) if progress else None
    if pbar is not None:
        widen_indeterminate_notebook_bar(pbar)
    try:
        with pq.ParquetWriter(tmp_parquet, schema, compression='zstd') as writer:
            with lzma.open(input_file, 'rt') as fh:
                chunk: list[tuple[str, dict]] = []
                record_lines: list[str] = []
                for line in fh:
                    if line.startswith(_SD_RECORD_END):
                        ctab, props = _split_ctab_and_properties(record_lines)
                        record_lines = []
                        if ctab.strip():
                            chunk.append((ctab, props))
                        if len(chunk) == chunksize:
                            _write_chunk(writer, chunk)
                            chunk = []
                    else:
                        record_lines.append(line)
                # A malformed file missing the final '$$$$' still yields its last record.
                if record_lines:
                    ctab, props = _split_ctab_and_properties(record_lines)
                    if ctab.strip():
                        chunk.append((ctab, props))
                if chunk:
                    _write_chunk(writer, chunk)
        tmp_parquet.replace(output_file)
    finally:
        if pbar is not None:
            pbar.close()
        tmp_parquet.unlink(missing_ok=True)


def convert_gz_to_xz(
        input_file: str | Path,
        output_file: str | Path,
        compression_level: int | None = lzma.PRESET_DEFAULT,
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
            widen_indeterminate_notebook_bar(pbar)
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
