# -*- coding: utf-8 -*-

"""Download utilities of the Papyrus scripts."""

import multiprocessing as mp
import queue
import shutil
import textwrap
import unicodedata
import warnings
import zipfile
from pathlib import Path
from typing import cast

import pystow
import requests
from tqdm.auto import tqdm

from .utils.IO import (
    PapyrusVersion,
    _set_root_folder,
    assert_sha256sum,
    convert_sd_to_parquet,
    convert_xz_to_parquet,
    enough_disk_space,
    get_disk_space,
    get_papyrus_links,
    load_data_type_schemas,
    new_session,
    notebook_safe_ncols,
    read_jsonfile,
    widen_indeterminate_notebook_bar,
    write_jsonfile,
)

# Download / integrity constants
_CHUNKSIZE = 1_048_576  # 1 MB per streaming chunk
_RETRIES = 3

#: Maps a logical file-type key to the section of data_types.json holding
#: its column dtypes, for file types whose dtypes are known ahead of time.
#: Absent keys (e.g. 'proteins', 'proteins_prodec') convert with inferred
#: dtypes instead.
_SCHEMA_KEY_BY_FTYPE = {
    'papyrus++':       'papyrus',
    '2D_papyrus':      'papyrus',
    '3D_papyrus':      'papyrus',
    '2D_mold2':        'mold2',
    '2D_cddd':         'CDDD',
    '2D_mordred':      'mordred_2D',
    '3D_mordred':      'mordred_3D',
    '2D_fingerprint':  'ECFP6',
    '3D_fingerprint':  'E3FP',
    'proteins_unirep': 'unirep',
}

#: File types whose reader keeps empty-string cells distinct from nulls
#: (reader.read_protein_set passes null_values=[] for this reason) - the
#: Parquet conversion must apply the same null-value handling, since once a
#: cell is written as a genuine Parquet null the empty-string/null
#: distinction can no longer be recovered at read time.
_NULL_VALUES_BY_FTYPE: dict[str, list] = {'proteins': []}

#: Maps a logical file-type key to the data_size.json key holding its
#: (naive, physical-line) row count - used only to give the per-file
#: conversion progress bar an approximate total to show; a missing key
#: (e.g. this Papyrus release predates that entry) just means no total is
#: shown, falling back to an open-ended running count.
_SIZE_KEY_BY_FTYPE = {
    'papyrus++':       'papyrus_++',
    '2D_papyrus':      'papyrus_2D',
    '3D_papyrus':      'papyrus_3D',
    'proteins':        'papyrus_proteins',
    '2D_mold2':        'mold2',
    '2D_cddd':         'CDDD',
    '2D_mordred':      'mordred_2D',
    '3D_mordred':      'mordred_3D',
    '2D_fingerprint':  'ECFP6',
    '3D_fingerprint':  'E3FP',
    'proteins_unirep': 'unirep',
    'proteins_prodec': 'prodec',
    '2D_structures':   'structures_2D',
    '3D_structures':   'structures_3D',
}

#: Fixed progress-bar width (characters) for the download and conversion
#: bars, so none of them grows to fill an entire wide terminal.
_PBAR_NCOLS = 100

#: Fixed width (characters) for the boxed notice/disclaimer/error banners
#: printed by download_papyrus - kept in the same spirit as _PBAR_NCOLS.
_BANNER_WIDTH = 78


def _display_width(text: str) -> int:
    """Approximate the terminal column width of *text*.

    ``len()`` counts an emoji as a single character, but most terminals
    render it two columns wide - padding banner rows with ``.ljust()``
    directly therefore misaligns any row containing one relative to plain-
    text rows. Unicode's East Asian Width property ('W'ide/'F'ullwidth)
    reliably tags emoji this way too, without needing a wcwidth dependency.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1 for ch in text)


def _ljust_display(text: str, width: int) -> str:
    """Right-pad *text* with spaces until it occupies *width* display columns."""
    return text + ' ' * max(0, width - _display_width(text))


def _print_banner(emoji: str, title: str, *lines: str) -> None:
    """Print a boxed, emoji-tagged banner for a user-facing notice/error.

    :param emoji: single leading emoji tagging the banner's kind (info,
        warning, error, ...)
    :param title: short header line (assumed to already fit - unlike
        *lines*, never dynamically long, so not wrapped)
    :param lines: body lines, word-wrapped to fit the banner's fixed width
        (e.g. a dynamically-built list of stale versions can otherwise run
        arbitrarily long and overflow past the border)
    """
    inner_width = _BANNER_WIDTH - 2
    text_width = _BANNER_WIDTH - 3  # inner_width, minus the leading space after '│'
    rule = '─' * inner_width

    def _row(text: str) -> str:
        return f'│ {_ljust_display(text, text_width)}│'

    print(f'┌{rule}┐')
    print(_row(f'{emoji} {title}'))
    if lines:
        print(f'├{rule}┤')
        for line in lines:
            for wrapped in textwrap.wrap(line, text_width) or ['']:
                print(_row(wrapped))
    print(f'└{rule}┘')


def _parquet_sibling(fpath: Path) -> Path | None:
    """Return the ``.parquet`` sibling of a downloaded ``.tsv.xz``/``.sd.xz`` file.

    None if *fpath* is not a convertible file - non-data files (zips, JSON,
    text) are never converted.
    """
    name = fpath.name.lower()
    if name.endswith('.tsv.xz') or name.endswith('.sd.xz'):
        return fpath.with_suffix('.parquet')
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_versions(
    version: str | list[str],
    files: dict,
    all_revisions: bool = False,
) -> list[PapyrusVersion]:
    """Normalise the *version* argument into a sorted, deduplicated list of :class:`PapyrusVersion` objects.

    Accepts any combination of new-format canonical strings (``'2022.04.2'``),
    new-format alias strings (``'2022.04'``), old-format strings (``'05.4'``),
    ``'latest'``, and ``'all'``.

    :param version: raw version argument passed by the caller
    :param files: the links dictionary returned by :func:`get_papyrus_links`,
        used to enumerate available versions
    :param all_revisions: when True, ``'all'`` expands to every revision present
        in *files*; a two-part alias string expands to all its revisions.
        When False (default), ``'all'`` expands to only the latest revision of
        each version alias, and a two-part alias resolves to its latest revision.
    :raises ValueError: if an unrecognised version string is supplied
    """
    _aliases = PapyrusVersion.aliases

    # Derive the latest version from the local aliases table — no network call.
    _latest_alias = _aliases['alias'].max()
    _latest_rev = _aliases[_aliases['alias'] == _latest_alias]['revision'].max()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        latest_pv = PapyrusVersion(version=f'{_latest_alias}.{_latest_rev}')

    # Canonical keys only (skip legacy '05.X'-style keys for user-facing messages).
    canonical_keys = [k for k in files if k.count('.') == 2]

    if not isinstance(version, list):
        version = [version]

    resolved: list[PapyrusVersion] = []
    for v in version:
        if v == 'latest':
            resolved.append(latest_pv)

        elif v == 'all':
            if all_revisions:
                # Every canonical key in links.json that aliases.json can resolve.
                for key in canonical_keys:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            resolved.append(PapyrusVersion(version=key))
                    except ValueError:
                        warnings.warn(
                            f'{key!r} is in links.json but has no entry in aliases.json; skipping.',
                            UserWarning,
                            stacklevel=3,
                        )
            else:
                # Default: latest revision of each known alias only.
                latest_rows = (
                    _aliases
                    .assign(_r=_aliases['revision'].astype(int))
                    .sort_values('_r')
                    .groupby('alias', sort=False)
                    .last()
                    .reset_index()
                )
                for _, row in latest_rows.iterrows():
                    canonical = f"{row['alias']}.{row['revision']}"
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        pv = PapyrusVersion(version=canonical)
                    if pv.version in files or pv.version_old_fmt in files:
                        resolved.append(pv)

        elif all_revisions and v.count('.') < 2:
            # Two-part alias or old-format string: expand all known revisions.
            # Resolve the two-part string to its canonical alias (e.g. '2022.04').
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _probe = PapyrusVersion(version=v)
                alias = _probe._version
            except ValueError:
                alias = v  # try as-is; will fail at the key-check below if invalid
            # Use links.json as the authoritative list of available revisions for
            # this alias (aliases.json only carries the latest revision per alias).
            prefix = alias + '.'
            matching_keys = sorted(
                [k for k in canonical_keys if k.startswith(prefix)],
                key=lambda k: int(k.split('.')[-1]),
            )
            if not matching_keys:
                valid = ['latest', 'all'] + canonical_keys
                raise ValueError(
                    f'version must be one of [{", ".join(valid)}], got {v!r}',
                )
            for key in matching_keys:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pv = PapyrusVersion(version=key)
                resolved.append(pv)

        else:
            try:
                pv = PapyrusVersion(version=v)
            except ValueError:
                valid = ['latest', 'all'] + canonical_keys
                raise ValueError(
                    f'version must be one of [{", ".join(valid)}], got {v!r}',
                ) from None
            if pv.version not in files and pv.version_old_fmt not in files:
                valid = ['latest', 'all'] + canonical_keys
                raise ValueError(
                    f'version must be one of [{", ".join(valid)}], got {v!r}',
                )
            resolved.append(pv)

    # Deduplicate while preserving sort order (oldest first).
    seen: set = set()
    unique: list[PapyrusVersion] = []
    for pv in sorted(set(resolved)):
        if pv not in seen:
            seen.add(pv)
            unique.append(pv)
    return unique


def _file_path(papyrus_version_root: pystow.Module, ftype: str, fname: str) -> Path:
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
        return papyrus_version_root.join(name=fname)
    if ftype in _STRUCTURE_FTYPES:
        return papyrus_version_root.join('structures', name=fname)
    return papyrus_version_root.join('descriptors', name=fname)


def _iter_entries(ftype_data) -> list[dict]:
    """Return *ftype_data* as a list of file-entry dicts.

    Normalises the raw links-JSON value, which may be a single dict or a
    list of dicts, to always a list.

    :param ftype_data: value from the links JSON for a given file type
    :raises ValueError: if the value is neither a dict nor a list
    """
    if isinstance(ftype_data, dict):
        return [ftype_data]
    if isinstance(ftype_data, list):
        return ftype_data
    raise ValueError(
        f'Papyrus links file corrupted: expected dict or list, '
        f'got {type(ftype_data).__name__!r}.',
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
    json_file = papyrus_root.join(name='versions.json')
    # versions.json is always written as a JSON array (see the write_jsonfile call below).
    existing = cast(list, read_jsonfile(json_file)) if json_file.is_file() else []
    path_key = pv.pystow_path_key
    if add:
        updated = sorted(set(existing + [path_key]))
    else:
        updated = sorted(v for v in existing if v != path_key)
    write_jsonfile(updated, json_file)


def _get_version_files(files: dict, pv: PapyrusVersion) -> dict:
    """Return the file-metadata sub-dict for *pv* from the links dictionary.

    Tries the canonical key (``pv.version``, e.g. ``'2022.04.2'``) first,
    then falls back to the old-format key (``pv.version_old_fmt``, e.g.
    ``'05.4'``) for transitional ``links.json`` files that still carry both
    key styles.  Emits a :class:`DeprecationWarning` when the old-format key
    is used or when the entry itself carries a ``'__deprecated__'`` field.

    :param files: full links dict from :func:`~papyrus_scripts.utils.IO.get_papyrus_links`
    :param pv: version whose file metadata to retrieve
    :raises KeyError: if neither the canonical nor the old-format key is found
    """
    if pv.version in files:
        entry = files[pv.version]
    elif pv.version_old_fmt in files:
        warnings.warn(
            f"links.json uses the legacy key {pv.version_old_fmt!r}. "
            f"Upgrade papyrus-scripts to use the new key {pv.version!r}.",
            DeprecationWarning,
            stacklevel=3,
        )
        entry = files[pv.version_old_fmt]
    else:
        raise KeyError(
            f"No download links found for version {pv.version!r} "
            f"(also tried legacy key {pv.version_old_fmt!r}).",
        )
    msg = entry.get('__deprecated__')
    if msg:
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
    return entry


#: Sentinel placed on the conversion queue to signal "no more files are
#: coming, exit normally". A real task is always a dict, so plain ``None``
#: can never be mistaken for it - and unlike a fresh ``object()``, ``None``
#: survives a multiprocessing.Queue's pickle/unpickle round-trip as the
#: exact same singleton, so `is` comparisons against it still work on the
#: worker side of the process boundary (a distinct sentinel object does
#: not: pickling it in put() and unpickling it in get() produces a new,
#: unequal object each time, silently breaking an `is` check like this
#: one's).
_CONVERSION_DONE = None


#: Messages put on the progress queue by _convert_worker, consumed by
#: _drain_progress_queue in the parent process:
#: ('start', desc, total_rows) - new file: reset row count, set desc/total
#: ('reset',)   - dtype drift healing: undo credited rows, reset row count
#: ('chunk', n) - n more rows written; advances the row-unit progress bar
#: ('done',)    - current file finished
_PROGRESS_START = 'start'
_PROGRESS_RESET = 'reset'
_PROGRESS_CHUNK = 'chunk'
_PROGRESS_DONE = 'done'


def _convert_worker(
        task_queue: mp.Queue,
        error_queue: mp.Queue,
        progress_queue: mp.Queue,
        progress: bool,
) -> None:
    """Consume Parquet-conversion tasks from *task_queue*, one at a time, until the sentinel is read.

    Runs in its own process so CPU-bound Parquet conversion overlaps with
    the main process' network downloads instead of blocking them - the main
    process pushes one task per tabular file as soon as that file finishes
    downloading, then a final :data:`_CONVERSION_DONE` sentinel once every
    file has been handled, at which point this process exits normally.

    Renders no progress bars itself - reports progress via *progress_queue*
    instead (see the module-level ``_PROGRESS_*`` constants for the message
    protocol), for the parent process to render. Two independent processes
    each rendering their own position-pinned tqdm bars (this worker's
    former approach, alongside the parent's own download-progress bar) was
    tried and reverted: even with a lock shared between them (tqdm's
    documented pattern for coexisting bars across processes, which does
    prevent byte-level interleaving/garbling of a single write), each
    process' cursor-movement math is still computed independently, with no
    shared notion of the terminal's actual current state - so one
    process's bar redraw can scroll the terminal in a way the other has no
    way of knowing about, and stray blank lines accumulate over time. A
    single process owning every bar (tqdm's well-supported case) doesn't
    have this failure mode at all.

    Each task is a ``dict`` of keyword arguments for
    :func:`~papyrus_scripts.utils.IO.convert_xz_to_parquet` (``.tsv.xz``) or
    :func:`~papyrus_scripts.utils.IO.convert_sd_to_parquet` (``.sd.xz``,
    dispatched on ``fpath``'s suffix), plus the
    ``.xz`` original's path (``'fpath'``) - deleted once its conversion
    succeeds, matching the previous single-process behaviour. Any exception
    aborts the current file - including ``KeyboardInterrupt``: hitting
    Ctrl+C delivers ``SIGINT`` to every process in the group, so this one
    receives it too, independently of the parent. ``convert_xz_to_parquet``
    writes under a uniquely-named ``.converting`` temp file and only renames
    it to the final name on success; a half-written one left behind by this
    failure is cleaned up by that same function's own next attempt (not
    here), before this process reports the failure back to the parent via
    *error_queue* and exits - it does not keep processing further tasks
    after a failure.

    :param task_queue: FIFO of conversion tasks, fed by the parent process
    :param error_queue: used to report back a single value once this
        process is about to exit: ``None`` for a clean run, or a short
        string describing the exception that aborted it
    :param progress_queue: conversion-progress messages for the parent to
        render (see the ``_PROGRESS_*`` message protocol above); unused
        when *progress* is False
    :param progress: whether to report progress at all - when False, no
        messages are put on *progress_queue*, avoiding the per-chunk
        IPC/pickling cost of reporting progress nobody will render
    """
    try:
        while True:
            task = task_queue.get()
            if task is _CONVERSION_DONE:
                error_queue.put(None)
                return
            fpath: Path = task['fpath']
            parquet_path: Path = task['parquet_path']
            if progress:
                progress_queue.put((_PROGRESS_START, task['desc'], task['total_rows']))
            try:
                on_progress = (
                    (lambda n: progress_queue.put((_PROGRESS_CHUNK, n)))
                    if progress else None
                )
                if fpath.name.lower().endswith('.sd.xz'):
                    convert_sd_to_parquet(fpath, parquet_path, on_progress=on_progress)
                else:
                    convert_xz_to_parquet(
                        fpath, parquet_path,
                        separator='\t',
                        schema_overrides=task['schema_overrides'],
                        null_values=task['null_values'],
                        on_progress=on_progress,
                        on_reset=(
                            (lambda: progress_queue.put((_PROGRESS_RESET,)))
                            if progress else None
                        ),
                    )
                fpath.unlink()
            except BaseException as exc:
                error_queue.put(f'{type(exc).__name__}: {exc}')
                return
            if progress:
                progress_queue.put((_PROGRESS_DONE,))
    except BaseException:
        # Make sure a crash/interrupt before even reaching the per-task
        # try/except above (e.g. task_queue.get() itself interrupted) still
        # reports something, rather than leaving the parent blocked forever
        # on error_queue.get().
        error_queue.put('worker exited unexpectedly')
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_papyrus(outdir: str | Path | None = None,
                     version: str | list[str] = 'latest',
                     nostereo: bool = True,
                     stereo: bool = False,
                     only_pp: bool = True,
                     structures: bool = False,
                     descriptors: str | list[str] | None = 'all',
                     progress: bool = True,
                     disk_margin: float = 0.10,
                     update_links: bool = True,
                     all_revisions: bool = False,
                     keep_xz: bool = False) -> None:
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
    :param all_revisions: when False (default), ``'all'`` and two-part alias strings
        expand to only the latest revision of each version; when True, all revisions
        present in the links file are included
    :param keep_xz: by default (False), every downloaded tabular ``.tsv.xz``
        file (bioactivities, protein targets, descriptors) is losslessly
        converted to ``.parquet`` and the ``.xz`` original is deleted, so
        ``papyrus_scripts.reader`` can lazily scan Parquet instead of
        decompressing LZMA on every read. Set to True to keep the ``.xz``
        files as-is (e.g. if you intend to transcode them with
        :func:`~papyrus_scripts.utils.IO.convert_xz_to_gz` /
        :func:`~papyrus_scripts.utils.IO.convert_gz_to_xz`, which require
        the compressed originals to be present).
    """
    _set_root_folder(outdir)

    # Warn if any data downloaded with the old versioning format exists on disk.
    _versions_json = pystow.join('papyrus', name='versions.json')
    if _versions_json.is_file():
        _old_fmt_keys = set(PapyrusVersion.aliases['version'].values)
        _stale = [v for v in (read_jsonfile(_versions_json) or []) if v in _old_fmt_keys]
        if _stale:
            version_flags = ' '.join(f'--version {v}' for v in _stale)
            _print_banner(
                '📢', 'Notice: new data folder layout',
                'Papyrus-scripts now uses a new folder layout that includes the',
                'revision number (e.g. 2022.04.2/ instead of 05.4/).',
                'Existing data downloaded with the previous format was found:',
                f'  {", ".join(_stale)}',
                'Please remove those old folders before downloading new data:',
                f'  papyrus clean {version_flags} --remove_version',
                f'(or remove_papyrus(version={_stale!r}, version_root=True) from Python)',
            )

    files = get_papyrus_links(offline=not update_links)
    versions = _resolve_versions(version, files, all_revisions=all_revisions)

    if descriptors is None:
        descriptors = []
    if not isinstance(descriptors, list):
        descriptors = [descriptors]

    papyrus_root = pystow.module('papyrus')
    session = new_session()

    for pv in versions:
        version_files = _get_version_files(files, pv)
        papyrus_version_root = pystow.module('papyrus', pv.pystow_path_key)

        # ------------------------------------------------------------------
        # Build the set of logical file-type keys to download
        # ------------------------------------------------------------------
        downloads: set = {'readme', 'requirements', 'proteins'}

        if nostereo:
            downloads.add('papyrus++')
            if not only_pp:
                downloads.add('2D_papyrus')
            elif progress:
                _print_banner(
                    '💡', 'Downloading the Papyrus++ subset',
                    'You are downloading the high-quality Papyrus++ dataset.',
                    'Should you want to access the entire, though of lower quality,',
                    'Papyrus dataset, look into additional switches of this command.',
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
                warnings.warn(
                    f'ProDEC descriptors are not available for Papyrus version {pv}. Skipping.',
                    stacklevel=2,
                )

        # Drop any key that is absent from this version's link table
        downloads = {ft for ft in downloads if ft in version_files}

        # ------------------------------------------------------------------
        # Check available disk space
        # ------------------------------------------------------------------
        total = _total_size(downloads, version_files)

        if progress:
            print(f'📦 {len(downloads)} file(s) to download - {tqdm.format_sizeof(total)}B total')

        base = papyrus_version_root.base
        free = get_disk_space(base)
        remaining = free - total
        if remaining <= 0:
            _print_banner(
                '🚫', 'Not enough disk space',
                'There would be no space left on disk after this download.',
                f'Available: {tqdm.format_sizeof(free)}B',
                f'Required:  {tqdm.format_sizeof(total)}B',
            )
            return
        if not enough_disk_space(base, total, disk_margin):
            _print_banner(
                '⚠️', 'Low disk space after download',
                f'Only {disk_margin:.0%} of free space is normally kept as a safety margin.',
                f'Available: {tqdm.format_sizeof(free)}B',
                f'Required:  {tqdm.format_sizeof(total)}B',
                f'Remaining after download: {tqdm.format_sizeof(remaining)}B',
            )
            confirmation = input('Continue with the download anyway (Y/N): ')
            if confirmation != 'Y':
                print('Download was aborted.')
                return

        # ------------------------------------------------------------------
        # Download (each tabular file is converted to Parquet immediately
        # after it finishes downloading, rather than only once every file
        # has been downloaded - 'requirements' (the zip holding
        # data_types.json) is processed first so the schema it provides is
        # already on disk by the time the first tabular file needs it.
        # ------------------------------------------------------------------
        ordered_ftypes = sorted(downloads, key=lambda ft: ft not in ('readme', 'requirements'))

        if progress:
            pbar = tqdm(
                total=total,
                desc=f'Downloading version {pv.version}',
                unit='B',
                unit_scale=True,
                ncols=notebook_safe_ncols(_PBAR_NCOLS),
                position=0,
            )

        schemas: dict = {}
        sizes: dict = {}
        metadata_loaded = False

        # Parquet conversion runs in its own process, fed one task per
        # tabular file via a bounded FIFO queue (maxsize=1: the main
        # process may get at most one file ahead of the converter, instead
        # of piling up every .xz original on disk before any of them is
        # converted) - so downloading file N+1 overlaps with converting
        # file N rather than the two happening strictly one after another.
        # The worker itself renders no progress bars - it reports progress
        # via progress_queue instead (see _convert_worker's docstring), and
        # both bars below are rendered by this (parent) process only - one
        # process owning every bar it displays is tqdm's well-supported
        # case for simultaneous position-pinned bars (see
        # _convert_worker's docstring for the multi-*process* rendering
        # that was tried and reverted; that failure mode doesn't apply
        # here). converting_pbar (position=1) is created up front and kept
        # live for the whole download+conversion overlap, right below the
        # download bar (position=0), instead of staying hidden until
        # downloading finishes.
        # noqa: B023 below - these and the nested functions referencing them
        # are (re)created fresh every 'pv' iteration and only ever called
        # synchronously within that same iteration, never stored for later,
        # so the values seen are always current, not a stale closure.
        converter_process: mp.Process | None = None
        task_queue: mp.Queue | None = None
        error_queue: mp.Queue | None = None
        progress_queue: mp.Queue | None = None
        converting_pbar: tqdm | None = None
        current_desc = ''
        current_rows = 0
        current_total: int | None = None

        def _update_current_file_description() -> None:
            # Only ever called once converting_pbar has been created (see call sites).
            if converting_pbar is None:  # noqa: B023
                raise RuntimeError('converting_pbar not created despite progress=True')
            total_str = f'{current_total:,}' if current_total is not None else '?'  # noqa: B023
            converting_pbar.set_description(f'{current_desc} ({current_rows:,}/{total_str} rows)')  # noqa: B023

        def _mark_current_file_complete() -> None:
            # Only ever called once converting_pbar has been created (see call sites).
            if converting_pbar is None:  # noqa: B023
                raise RuntimeError('converting_pbar not created despite progress=True')
            # refresh=False: caller's own refresh() follows right after; two
            # back-to-back refreshes can leave the cursor off by a line on Windows.
            converting_pbar.set_description(f'{current_desc} (complete)', refresh=False)  # noqa: B023

        def _drain_progress_queue() -> None:
            """Apply every conversion-progress message available right now, without blocking."""
            nonlocal current_desc, current_rows, current_total
            # Only ever called once progress_queue has been created (see call sites).
            if progress_queue is None:  # noqa: B023
                raise RuntimeError('progress_queue not created despite progress=True')
            while True:
                try:
                    message = progress_queue.get_nowait()  # noqa: B023
                except queue.Empty:
                    return
                kind = message[0]
                if kind == _PROGRESS_START:
                    _, current_desc, current_total = message
                    current_rows = 0
                elif kind == _PROGRESS_RESET:
                    # Undo rows already credited before restarting from 0.
                    if converting_pbar is not None and current_rows:  # noqa: B023
                        converting_pbar.update(-current_rows)  # noqa: B023
                    current_rows = 0
                elif kind == _PROGRESS_CHUNK:
                    current_rows += message[1]
                # Defensive only: progress_queue only ever receives messages
                # when progress is True, which is also when converting_pbar
                # is created - the two are never out of sync in practice.
                if converting_pbar is None:  # pragma: no cover  # noqa: B023
                    continue
                if kind == _PROGRESS_CHUNK:
                    # update()'s refresh also repaints the description set below.
                    converting_pbar.update(message[1])  # noqa: B023
                    _update_current_file_description()
                elif kind == _PROGRESS_DONE:
                    _mark_current_file_complete()
                    converting_pbar.refresh()  # noqa: B023
                elif kind in (_PROGRESS_START, _PROGRESS_RESET):
                    _update_current_file_description()

        def _wait_for_converter() -> None:
            """Block until the converter process exits, draining progress messages meanwhile."""
            # Only ever called once converter_process has been started (see call sites).
            if converter_process is None:  # noqa: B023
                raise RuntimeError('converter_process not started')
            while converter_process.is_alive():  # noqa: B023
                _drain_progress_queue()
                converter_process.join(timeout=0.1)  # noqa: B023
            _drain_progress_queue()
            if converting_pbar is not None:  # noqa: B023
                converting_pbar.close()  # noqa: B023

        def _enqueue(item) -> None:
            """Put *item* on task_queue, staying responsive while backpressured.

            task_queue's maxsize=1 means this blocks whenever the converter
            hasn't yet taken the previous task - which can be a long wait
            for a large file. A plain blocking put() here previously gave
            zero feedback during that wait, indistinguishable from a
            genuine hang. Retrying with a short timeout instead lets each
            iteration drain the progress queue, which keeps converting_pbar
            live throughout the wait.

            A dead converter never drains task_queue again, so liveness is
            checked on every timeout instead of retrying forever.
            """
            if task_queue is None:  # noqa: B023
                raise RuntimeError('task_queue not created')
            while True:
                try:
                    task_queue.put(item, timeout=0.2)  # noqa: B023
                    return
                except queue.Full:
                    if converter_process is not None and not converter_process.is_alive():  # noqa: B023
                        if error_queue is None:  # noqa: B023
                            raise RuntimeError('error_queue not created') from None
                        try:
                            error = error_queue.get(timeout=1.0)  # noqa: B023
                        except queue.Empty:
                            error = 'worker exited unexpectedly'
                        raise RuntimeError(f'Parquet conversion failed: {error}') from None
                    _drain_progress_queue()

        if not keep_xz:
            # ftype per file still needing conversion; used once data_size.json
            # is read to compute converting_pbar's total row count.
            to_convert_ftypes: list[str] = []
            for ftype in ordered_ftypes:
                for entry in _iter_entries(version_files[ftype]):
                    fpath = _file_path(papyrus_version_root, ftype, entry['name'])
                    parquet_path = _parquet_sibling(fpath)
                    if parquet_path is not None and not parquet_path.is_file():
                        to_convert_ftypes.append(ftype)

            task_queue = mp.Queue(maxsize=1)
            error_queue = mp.Queue()
            progress_queue = mp.Queue()
            if progress:
                converting_pbar = tqdm(
                    # Unknown total until sizes is loaded, unless nothing to convert.
                    total=(0 if not to_convert_ftypes else None),
                    desc='Converting files', unit=' row', unit_scale=True,
                    ncols=notebook_safe_ncols(_PBAR_NCOLS), position=1,
                )
                widen_indeterminate_notebook_bar(converting_pbar)
            converter_process = mp.Process(
                target=_convert_worker,
                args=(task_queue, error_queue, progress_queue, progress),
                daemon=True,
            )
            converter_process.start()

        try:
            for ftype in ordered_ftypes:
                for entry in _iter_entries(version_files[ftype]):
                    dname = entry['name']
                    durl = entry['url']
                    dsize = entry['size']
                    dhash = entry['sha256']
                    fpath = _file_path(papyrus_version_root, ftype, dname)

                    # Skip entirely if already converted to Parquet by a
                    # previous call (the .xz original was deliberately
                    # deleted in that case).
                    parquet_path = _parquet_sibling(fpath)
                    if parquet_path is not None and parquet_path.is_file():
                        if progress:
                            pbar.update(dsize)
                        continue

                    # Download unless already present and intact.
                    if fpath.is_file() and assert_sha256sum(fpath, dhash):
                        if progress:
                            pbar.update(dsize)
                    else:
                        # Attempt download with up to _RETRIES tries
                        success = False
                        remaining = _RETRIES
                        last_error: Exception | None = None
                        while not success and remaining > 0:
                            try:
                                res = session.get(
                                    durl,
                                    stream=True,
                                    verify=True,
                                )
                                with open(fpath, 'wb') as fh:
                                    for chunk in res.iter_content(chunk_size=_CHUNKSIZE):
                                        fh.write(chunk)
                                        if progress:
                                            pbar.update(len(chunk))
                                        if progress_queue is not None:
                                            _drain_progress_queue()
                                success = assert_sha256sum(fpath, dhash)
                                last_error = None
                            except requests.exceptions.RequestException as err:
                                # Chunk failed mid-stream - retry like a hash mismatch.
                                success = False
                                last_error = err

                            if not success:
                                remaining -= 1
                                # Never leave a truncated file behind.
                                fpath.unlink(missing_ok=True)
                                if progress:
                                    if last_error is not None:
                                        reason = f'Connection error downloading {dname}: {last_error}. '
                                    else:
                                        reason = f'SHA256 mismatch for {dname}. '
                                    msg = (
                                            reason
                                            + (f'Retrying ({remaining} left).'
                                               if remaining > 0
                                               else f'All {_RETRIES} attempts failed.')
                                    )
                                    pbar.write(msg)

                        if not success:
                            # pbar is intentionally left open here (not
                            # closed) - see the except handler below for why.
                            raise OSError(f'Download failed for {dname}') from last_error

                    # Extract ZIP archives in-place
                    if dname.endswith('.zip'):
                        dest = fpath.parent
                        with zipfile.ZipFile(fpath) as zh:
                            for name in zh.namelist():
                                zh.extract(name, dest)
                        fpath.unlink()
                        continue

                    # Queue this tabular file for conversion right away
                    # rather than waiting for every other file to finish
                    # downloading first.
                    if not keep_xz and parquet_path is not None:
                        if not metadata_loaded:
                            dtype_file = papyrus_version_root.join(name='data_types.json')
                            if dtype_file.is_file():
                                schemas = load_data_type_schemas(papyrus_version_root)
                                # data_size.json is always written as a JSON object.
                                sizes = cast(dict, read_jsonfile(papyrus_version_root.join(name='data_size.json')))
                                metadata_loaded = True
                                # Sum per-ftype row counts into the real total,
                                # unless any is unknown (avoid understating it).
                                if progress and converting_pbar is not None and to_convert_ftypes:
                                    per_file_rows = [
                                        sizes.get(_SIZE_KEY_BY_FTYPE.get(ft))
                                        for ft in to_convert_ftypes
                                    ]
                                    if all(n is not None for n in per_file_rows):
                                        converting_pbar.total = sum(per_file_rows)
                                        converting_pbar.refresh()
                        if task_queue is None:
                            raise RuntimeError('task_queue not created')
                        _enqueue({
                            'fpath': fpath,
                            'parquet_path': parquet_path,
                            'schema_overrides': schemas.get(_SCHEMA_KEY_BY_FTYPE.get(ftype)),
                            'null_values': _NULL_VALUES_BY_FTYPE.get(ftype),
                            'total_rows': sizes.get(_SIZE_KEY_BY_FTYPE.get(ftype)),
                            # ftype (e.g. 'papyrus++', '2D_mold2') rather
                            # than fpath.name: the real filenames (e.g.
                            # '05.6++_combined_set_without_stereochemistry
                            # .tsv') are long enough to overflow the fixed
                            # bar width (_PBAR_NCOLS) on their own, before
                            # the bar itself even starts.
                            'desc': f'Converting {ftype}',
                        })
        except BaseException:
            # Downloading failed (or was interrupted) - still let the
            # converter process drain and exit cleanly instead of leaving
            # it running in the background, but propagate the original
            # failure rather than whatever it reports back, if anything.
            # pbar.close() must come *after* converting_pbar has finished
            # and closed, not before: tqdm's close() (with leave=True, our
            # default) writes an explicit, uncoordinated newline of its own
            # once the bar's own final state is drawn - a real newline that
            # a still-active position=1 bar has no way to account for in
            # its own cursor math, permanently shifting it one row off for
            # every redraw from that point on. Closing bottom-up (highest
            # position first) avoids this entirely.
            if converter_process is not None:
                if task_queue is None:
                    raise RuntimeError('task_queue not created') from None
                _enqueue(_CONVERSION_DONE)
                _wait_for_converter()
            if progress:
                pbar.close()
            raise

        if converter_process is not None:
            # Signal "no more files are coming" and wait for every already
            # -queued conversion to finish before this version is marked
            # as downloaded. Still before pbar.close() - see the except
            # handler's comment above for why.
            if task_queue is None:
                raise RuntimeError('task_queue not created')
            if error_queue is None:
                raise RuntimeError('error_queue not created')
            _enqueue(_CONVERSION_DONE)
            _wait_for_converter()
            error = error_queue.get()
            if error is not None:
                if progress:
                    pbar.close()
                raise RuntimeError(f'Parquet conversion failed: {error}')

        if progress:
            pbar.close()

        # Register this version in the local versions.json
        _update_versions_json(papyrus_root, pv, add=True)


def remove_papyrus(
        outdir: str | Path | None = None,
        version: str | list[str] = 'latest',
        papyruspp: bool = False,
        bioactivities: bool = False,
        proteins: bool = False,
        nostereo: bool = True,
        stereo: bool = False,
        structures: bool = False,
        descriptors: str | list[str] = 'all',
        other_files: bool = False,
        version_root: bool = False,
        papyrus_root: bool = False,
        force: bool = False,
        progress: bool = True,
        all_revisions: bool = False,
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
    :param all_revisions: when False (default), ``'all'`` and two-part alias
        strings expand to only the latest revision of each version; when True,
        all revisions present in the links file are included
    """
    _set_root_folder(outdir)

    files = get_papyrus_links()
    versions = _resolve_versions(version, files, all_revisions=all_revisions)

    if not isinstance(descriptors, list):
        descriptors = [descriptors]

    papyrus_root_mod = pystow.module('papyrus')

    # ------------------------------------------------------------------
    # Nuclear option: wipe everything
    # ------------------------------------------------------------------
    if papyrus_root:
        if not force:
            confirmation = input(
                'Confirm the removal of ALL Papyrus data and versions (Y/N): ',
            )
            if confirmation != 'Y':
                print('Removal was aborted.')
                return
        shutil.rmtree(papyrus_root_mod.base)
        if progress:
            print('All Papyrus data was successfully removed.')
        return

    for pv in versions:
        version_files = _get_version_files(files, pv)
        papyrus_version_root = pystow.module('papyrus', pv.pystow_path_key)

        # --------------------------------------------------------------
        # Per-version nuclear option: wipe one version folder
        # --------------------------------------------------------------
        if version_root:
            if not force:
                confirmation = input(
                    f'Confirm the removal of version {pv} of Papyrus data (Y/N): ',
                )
                if confirmation != 'Y':
                    print('Removal was aborted.')
                    continue
            shutil.rmtree(papyrus_version_root.base)
            if progress:
                print(f'Version {pv} of Papyrus was successfully removed.')
            _update_versions_json(papyrus_root_mod, pv, add=False)
            continue

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
        present: list[str] = []  # ftypes confirmed to exist on disk

        for ftype in list(removal):
            ftype_data = version_files[ftype]
            # Multi-entry types (chunked archives) are handled uniformly
            all_entries_exist = True
            ftype_size = 0
            for entry in _iter_entries(ftype_data):
                fpath = _file_path(papyrus_version_root, ftype, entry['name'])
                if fpath.is_file():
                    ftype_size += entry['size']
                else:
                    all_entries_exist = False
            if all_entries_exist:
                total += ftype_size
                present.append(ftype)

        if progress:
            print(
                f'Number of files to be removed: {len(present)}\n'
                f'Total size: {tqdm.format_sizeof(total)}B',
            )

        if not present:
            continue

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
                # Defensive only: every entry here was already confirmed
                # present just above: a file vanishing between that check
                # and here is an external race, not reachable in normal use.
                if not fpath.is_file():  # pragma: no cover
                    if progress:
                        pbar.update(entry['size'])
                    continue
                fpath.unlink()
                if progress:
                    pbar.update(entry['size'])

        if progress:
            pbar.close()

        # Update the local versions.json registry if all bioactivity data
        # is gone (conservative: only deregister when the root folder is empty)
        remaining_files = [
            f for f in papyrus_version_root.base.iterdir()
            if not f.name.startswith('.')
        ]
        if not remaining_files:
            _update_versions_json(papyrus_root_mod, pv, add=False)
