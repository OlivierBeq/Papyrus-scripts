# -*- coding: utf-8 -*-

"""Substructure and similarity search over the Papyrus dataset."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import multiprocessing.synchronize
import queue
import warnings
from abc import ABC
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pystow
import rdkit
from rdkit import Chem
from rdkit.Chem.rdSubstructLibrary import CachedSmilesMolHolder, PatternHolder, SubstructLibrary
from tqdm.auto import tqdm

try:
    import tables as tb
    HAS_TABLES = True  # pragma: no cover - exercised only with pytables installed
except ImportError:
    HAS_TABLES = False

try:
    # This whole block only runs with FPSim2 installed; every line below the
    # first import is unreachable otherwise (the first import raises first).
    from FPSim2.base import BaseEngine  # pragma: no cover
    from FPSim2.FPSim2 import FPSim2Engine
    from FPSim2.FPSim2Cuda import FPSim2CudaEngine  # pragma: no cover
    from FPSim2.io.backends.base import BaseStorageBackend  # pragma: no cover
    from FPSim2.io.backends.pytables import (  # pragma: no cover
        BATCH_WRITE_SIZE,
        calc_popcnt_bins_pytables,
        create_schema,
    )
    from FPSim2.io.chem import load_molecule  # pragma: no cover
    HAS_FPSIM2 = True  # pragma: no cover
except ImportError:
    HAS_FPSIM2 = False
    # Stub classes so class definitions below do not fail at import time.
    # Using dedicated stubs (not `object`) avoids MRO conflicts when these are
    # mixed with other bases that already implicitly inherit from `object`.
    class BaseStorageBackend:  # type: ignore[no-redef]
        """Stub for FPSim2.io.backends.base.BaseStorageBackend when FPSim2 is absent."""

    class BaseEngine:  # type: ignore[no-redef]
        """Stub for FPSim2.base.BaseEngine when FPSim2 is absent."""

    class FPSim2Engine:  # type: ignore[no-redef]
        """Stub for FPSim2.FPSim2.FPSim2Engine when FPSim2 is absent."""

    class FPSim2CudaEngine:  # type: ignore[no-redef]
        """Stub for FPSim2.FPSim2Cuda.FPSim2CudaEngine when FPSim2 is absent."""
    BATCH_WRITE_SIZE = 32_000  # FPSim2's own default; only used for queue sizing here

from .fingerprint import Fingerprint, MorganFingerprint, get_fp_from_name
from .utils.IO import PapyrusVersion, _prefer_parquet, _set_root_folder, get_num_rows_in_file, locate_file
from .utils.mol_reader import MolSupplier

# find_spec detects cupy without importing it; the actual import is
# deferred to _ensure_cupy() so CPU-only usage never pays that cost.
HAS_CUPY = importlib.util.find_spec('cupy') is not None
cupy: ModuleType | None = None  # set by _ensure_cupy() on first use

# ---------------------------------------------------------------------------
# Parallel-pipeline tuning constants
# ---------------------------------------------------------------------------
# Queues are bounded so a full queue blocks put() naturally - Queue.qsize()
# isn't portable (raises NotImplementedError on macOS).

READER_BATCH_SIZE = 256
"""Molecules per reader->worker queue message."""

WORKER_IPC_BATCH_SIZE = 1_000
"""Rows a worker buffers before one push to fp_output_queue. Smaller than
BATCH_WRITE_SIZE so fp_writer gets steady traffic and coalesces several
IPC batches per PyTables flush."""

INPUT_QUEUE_MAXSIZE_BATCHES = 2
"""Per-shard input queue size, in READER_BATCH_SIZE units (double-buffer)."""

OUTPUT_QUEUE_MAXSIZE_BATCHES_PER_WORKER = 2
"""fp_output_queue size, per worker. Bounded (not unbounded) so
worker/writer backpressure is real instead of workers racing ahead."""

DEFAULT_N_SHARDS_SINGLE_PROCESS = 4
"""Shard count for _single_process_create(), which has no worker count to derive one from."""

DEFAULT_PATTERN_HOLDER_BITS = 2048
"""Default PatternHolder(numBits=...) - RDKit's own default.

A 200-molecule benchmark (512-8192 bits) showed no query-time difference
and only a modest size increase, but is too small to judge the prescreen's
false-positive tradeoff at real scale - kept at the default pending
re-validation on a realistically sized dataset."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_optional_deps() -> None:
    """Raise :exc:`ImportError` when *tables* or *FPSim2* are unavailable."""
    missing = []
    if not HAS_TABLES:
        missing.append('tables')
    if not HAS_FPSIM2:
        missing.append('FPSim2')
    if missing:
        raise ImportError(
            'Some required dependencies are missing:\n\t' + ', '.join(missing),
        )


def _pump_progress_queue(
        progress_queue: multiprocessing.Queue | None,
        pbar: tqdm | None,
        timeout: float = 0.1,
) -> None:
    """Update *pbar* from counts on *progress_queue*; no-op if ``None``.

    Blocks up to *timeout* for the first count, then drains the rest
    without blocking.
    """
    if progress_queue is None:
        return
    try:
        pbar.update(progress_queue.get(timeout=timeout))
    except queue.Empty:
        return
    while True:
        try:
            pbar.update(progress_queue.get_nowait())
        except queue.Empty:
            return


def _ensure_cupy() -> ModuleType:
    """Import cupy lazily, only when a CUDA engine is actually constructed."""
    global cupy
    if cupy is None:
        import cupy as _cupy
        cupy = _cupy
    return cupy


def _validate_fingerprints(
        fingerprint: Fingerprint | list[Fingerprint] | None,
) -> list[Fingerprint]:
    """Normalise and validate the *fingerprint* argument.

    Returns a non-empty list of :class:`~fingerprint.Fingerprint` instances.
    Falls back to all derived fingerprint types when *fingerprint* is ``None``.

    :param fingerprint: a single :class:`~fingerprint.Fingerprint`, a list
        thereof, or ``None`` to use every available fingerprint type
    :raises ValueError: if any element is not a :class:`~fingerprint.Fingerprint`
    """
    if fingerprint is None:
        # Fingerprint (unlike a leaf subclass) always has subclasses, so
        # derived() always returns a list here. Every concrete subclass
        # overrides __init__ with its own no-argument signature; mypy only
        # sees the abstract base's.
        return [fp() for fp in Fingerprint.derived()]  # type: ignore[union-attr,call-arg]
    if not isinstance(fingerprint, list):
        fingerprint = [fingerprint]
    for fp in fingerprint:
        if not isinstance(fp, Fingerprint):
            raise ValueError(f'{fp!r} is not a supported Fingerprint instance.')
    return fingerprint


def _fp_table_path(fp: Fingerprint) -> str:
    """Return the HDF5 path for the fps table of *fp*."""
    return f'/similarity_info/{repr(fp)}/fps'


def _shard_for_mol_id(mol_id: int, n_shards: int) -> int:
    """Return the deterministic substructure-library shard index for *mol_id*.

    Used by every build/append path so shard membership stays consistent
    and partitions stay disjoint.
    """
    return mol_id % n_shards


def _derive_connectivity(props: dict, rdmol: Chem.Mol) -> tuple[str, str]:
    """Extract or derive ``(connectivity, inchikey)`` from SD-file properties.

    When ``connectivity`` is absent it is derived from the first block of the
    InChIKey. When ``InChIKey`` is absent it is computed from the molecule.

    :param props: result of ``rdmol.GetPropsAsDict()``
    :param rdmol: the RDKit molecule (used as a fallback for InChIKey)
    """
    inchikey = props.get('InChIKey', '') or Chem.MolToInchiKey(rdmol) or ''
    connectivity = props.get('connectivity', '') or (inchikey.split('-')[0] if inchikey else '')
    return connectivity, inchikey


def _decode_bytes_df(df: pl.DataFrame) -> pl.DataFrame:
    """Decode bytes columns (``Binary``/``Object``) to ``str``."""
    def _decode_col(c: str) -> pl.Expr:
        if df.schema[c] == pl.Binary:
            return pl.col(c).cast(pl.Utf8)
        if df.schema[c] == pl.Object:
            return pl.col(c).map_elements(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x,
                return_dtype=pl.Utf8,
            )
        return pl.col(c)

    return df.select([_decode_col(c) for c in df.columns])


def _build_result_df(
        raw_results,
        get_mapping,
        score_col: str,
) -> pl.DataFrame:
    """Convert raw FPSim2 ``(id, score)`` pairs into a labelled DataFrame.

    :param raw_results: iterable of ``(mol_id, score)`` tuples returned by
        an FPSim2 search method
    :param get_mapping: callable that maps a list of integer ids to a
        DataFrame with Papyrus identifiers
    :param score_col: column name to assign to the score values
    :returns: a DataFrame with Papyrus identifiers and the score column, or
        an empty DataFrame with those column names when no results are found
    """
    # len(), not truthiness: raw_results is a numpy array; bool() on a
    # multi-element array raises ValueError.
    pairs = list(zip(*raw_results, strict=True)) if len(raw_results) else []
    if not pairs:
        return pl.DataFrame(schema={
            'idnumber': pl.Int64, 'connectivity': pl.Utf8,
            'InChIKey': pl.Utf8, score_col: pl.Float32,
        })
    ids, scores = list(pairs[0]), list(pairs[1])
    df = get_mapping(ids)
    return df.with_columns(pl.Series(score_col, scores))


def sort_db_file(filename: str | Path, verbose: bool = False) -> None:
    """Sort an FPSubSim2 HDF5 file by fingerprint popcount.

    Sorting enables the efficient popcount-range pruning used by FPSim2 during
    similarity searches. The operation rewrites the file via a temporary copy.

    :param filename: path to the ``.h5`` FPSubSim2 database
    :param verbose: show a progress bar
    """
    filename = Path(filename)
    tmp_filename = filename.with_name(filename.name + '_tmp')
    if tmp_filename.is_file():
        tmp_filename.unlink()
    filename.rename(tmp_filename)

    filters = tb.Filters(complib='blosc', complevel=1, shuffle=True, bitshuffle=True)
    stats: dict = {'groups': 0, 'leaves': 0, 'links': 0, 'bytes': 0, 'hardlinks': 0}

    with tb.open_file(tmp_filename, mode='r') as src:
        with tb.open_file(filename, mode='w') as dst:
            siminfo_group = dst.create_group(
                dst.root, 'similarity_info', 'Infos for similarity search',
            )
            simfp_groups = [g for g in src.walk_groups('/similarity_info/') if g._v_name]
            group_tables = [
                (simfp_group, list(src.iter_nodes(simfp_group, classname='Table')))
                for simfp_group in simfp_groups
            ]
            total = sum(len(tables) for _, tables in group_tables) + 1

            pbar = tqdm(total=total, desc='Optimizing FPSubSim2 file', disable=not verbose)
            for simfp_group, fp_tables in group_tables:
                dst_group = simfp_group._f_copy(
                    siminfo_group, recursive=False, filters=filters, stats=stats,
                )
                pbar.set_description(f'Optimizing {simfp_group._v_name}')
                for fp_table in fp_tables:
                    dst_fp_table = fp_table.copy(
                        dst_group,
                        fp_table.name,
                        filters=filters,
                        copyuserattrs=True,
                        overwrite=True,
                        stats=stats,
                        start=None, stop=None, step=None,
                        chunkshape='auto',
                        sortby='popcnt',
                        check_CSI=True,
                        propindexes=True,
                    )
                    popcnt_bins = calc_popcnt_bins_pytables(dst_fp_table, fp_table.attrs.length)
                    dst_fp_table.close()
                    popcounts = dst.create_vlarray(
                        dst_group, 'popcounts', tb.ObjectAtom(),
                        f'Popcounts of {dst_group._v_name}',
                    )
                    for x in popcnt_bins:
                        popcounts.append(x)
                    popcounts.close()
                    pbar.update(1)
                dst_group._f_close()
            siminfo_group._f_close()

            pbar.set_description('Optimizing remaining groups and arrays')
            for node in src.iter_nodes(src.root):
                if isinstance(node, tb.group.Group):
                    if isinstance(node, tb.group.RootGroup) or 'similarity_info' in str(node):
                        continue
                    node._f_copy(
                        dst.root, node._v_name,
                        overwrite=True, recursive=True,
                        filters=filters, stats=stats,
                    )
                else:
                    node.copy(dst.root, node._v_name, overwrite=True, stats=stats)
            pbar.update(1)
            pbar.close()

    tmp_filename.unlink()


# ---------------------------------------------------------------------------
# FPSubSim2 — database creation and loading
# ---------------------------------------------------------------------------

class FPSubSim2:
    """Create, load, and extend a multi-fingerprint similarity/substructure database.

    Backed by HDF5 (via PyTables). The database stores:

    * one or more fingerprint tables (for Tanimoto / Tversky similarity)
    * a serialised RDKit ``SubstructLibrary`` (for exact subgraph isomorphism)
    * a molecule-ID → ``(connectivity, InChIKey)`` mapping table
    """

    def __init__(self) -> None:
        """Create an unconfigured instance; call create_from_papyrus/create/load next."""
        _check_optional_deps()
        self.version: PapyrusVersion | None = None
        self.is3d: bool | None = None
        self.sd_file: Path | None = None
        self.h5_filename: Path | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def create_from_papyrus(
            self,
            is3d: bool = False,
            version: str | PapyrusVersion = 'latest',
            outfile: str | Path | None = None,
            fingerprint: Fingerprint | list[Fingerprint] | None = None,
            root_folder: str | Path | None = None,
            progress: bool = True,
            njobs: int = 1,
            n_shards: int | None = None,
            pattern_holder_bits: int = DEFAULT_PATTERN_HOLDER_BITS,
    ) -> None:
        """Create an FPSubSim2 database from the downloaded Papyrus SD file.

        :param is3d: use the stereochemistry-aware (3D) SD file
        :param version: Papyrus dataset version
        :param outfile: output ``.h5`` path; auto-generated when ``None``
        :param fingerprint: fingerprint(s) to store; defaults to all available
        :param root_folder: Papyrus data root directory
            (default: pystow's home directory)
        :param progress: display progress bars
        :param njobs: worker processes for parallel fingerprint computation
            (``-1`` = all logical cores, ``1`` = single-process)
        :param n_shards: substructure-library shard count; only usable for
            single-process builds (parallel builds use one shard per worker,
            so setting this with ``njobs > 1`` raises :exc:`ValueError`)
        :param pattern_holder_bits: bits for the substructure library's
            ``PatternHolder`` prescreen (size vs. query-speed tradeoff)
        """
        if fingerprint is None:
            fingerprint = MorganFingerprint()

        self.version = version if isinstance(version, PapyrusVersion) else PapyrusVersion(version=version)
        if not self.version.is_downloaded(root_folder=root_folder):
            raise ValueError(f'Version {self.version.version} not found. Did you download it first?')
        self.is3d = is3d

        _set_root_folder(root_folder)

        structure_dir = pystow.join(
            'papyrus', self.version.pystow_path_key, 'structures',
        )
        filenames = locate_file(
            structure_dir,
            rf'\d+\.\d+_combined_{3 if is3d else 2}D_set_'
            rf'with{"out" if not is3d else ""}_stereochemistry\.sd.*',
        )
        self.sd_file = _prefer_parquet(filenames)
        total = get_num_rows_in_file(
            filetype='structures',
            is3D=is3d,
            version=self.version,
            root_folder=root_folder,
        )
        self.create(
            sd_file=self.sd_file,
            outfile=outfile,
            fingerprint=fingerprint,
            total=total,
            progress=progress,
            njobs=njobs,
            n_shards=n_shards,
            pattern_holder_bits=pattern_holder_bits,
        )

    def create(
            self,
            sd_file: str | Path,
            outfile: str | Path | None = None,
            fingerprint: Fingerprint | list[Fingerprint] | None = None,
            progress: bool = True,
            total: int | None = None,
            njobs: int = 1,
            n_shards: int | None = None,
            pattern_holder_bits: int = DEFAULT_PATTERN_HOLDER_BITS,
    ) -> None:
        """Create an FPSubSim2 database from any SD file.

        :param sd_file: path to the SD file containing structures
        :param outfile: output ``.h5`` path; auto-generated when ``None``
        :param fingerprint: fingerprint(s) to store; defaults to all available
        :param progress: display progress bars
        :param total: molecule count for progress bar display
        :param njobs: worker processes (``-1`` = all, ``1`` = single-process)
        :param n_shards: substructure-library shard count; only usable for
            single-process builds (parallel builds use one shard per worker,
            so setting this with ``njobs > 1`` raises :exc:`ValueError`)
        :param pattern_holder_bits: bits for the substructure library's
            ``PatternHolder`` prescreen (size vs. query-speed tradeoff)
        :raises ValueError: if *njobs* is invalid, or if *n_shards* is given
            for a parallel build
        """
        self.sd_file = Path(sd_file)
        fingerprint = _validate_fingerprints(fingerprint)

        dim_tag = '3D' if self.is3d else '2D'
        version_str = str(self.version) if self.version is not None else 'custom'
        self.h5_filename = Path(outfile) if outfile is not None else Path(
            f'Papyrus_{version_str}_FPSubSim2_{dim_tag}.h5',
        )

        if not isinstance(njobs, int) or njobs < -1:
            raise ValueError('njobs must be -1 or a positive integer.')

        parallel = njobs not in (0, 1)
        if parallel:
            if n_shards is not None:
                raise ValueError(
                    'n_shards cannot be set explicitly for parallel builds '
                    '(njobs > 1 or njobs == -1): the shard count always '
                    'equals the worker count in that mode.',
                )
            n_workers = max(multiprocessing.cpu_count() - 2, 1) if njobs == -1 else max(njobs - 1, 1)
            resolved_n_shards = n_workers
        else:
            resolved_n_shards = n_shards if n_shards is not None else DEFAULT_N_SHARDS_SINGLE_PROCESS

        filters = tb.Filters()
        with tb.open_file(self.h5_filename, mode='w') as h5file:
            simil_group = h5file.create_group(
                h5file.root, 'similarity_info', 'Infos for similarity search',
            )
            subst_group = h5file.create_group(
                h5file.root, 'substructure_info', 'Infos for substructure search',
            )
            subst_group._v_attrs.n_shards = resolved_n_shards
            subst_group._v_attrs.format_version = 2
            subst_group._v_attrs.pattern_holder_bits = pattern_holder_bits
            for shard_idx in range(resolved_n_shards):
                shard_group = h5file.create_group(
                    subst_group, f'shard_{shard_idx}', f'Substructure search library shard {shard_idx}',
                )
                h5file.create_earray(
                    shard_group, 'substruct_lib', tb.UInt64Atom(), (0,),
                    f'Substructure search library shard {shard_idx}',
                )
                # GetMatches() returns 0-based positions, not idnumber (which
                # is 1-based and can have gaps); mol_ids[k] maps position k
                # back to the true idnumber.
                h5file.create_earray(
                    shard_group, 'mol_ids', tb.Int64Atom(), (0,),
                    f'idnumber for each AddMol() position in shard {shard_idx}',
                )
            h5file.create_table(
                h5file.root, 'mol_mappings',
                np.dtype([
                    ('idnumber', '<i8'),
                    ('connectivity', 'S14'),
                    ('InChIKey', 'S27'),
                ],
                ),
                'Molecular mappings',
                expectedrows=1_300_000,
                filters=filters,
            )
            # Store RDKit version, Papyrus version string, and stereo flag so
            # that load() can warn when the RDKit version has changed and can
            # reconstruct the PapyrusVersion.
            config = h5file.create_vlarray(h5file.root, 'config', atom=tb.ObjectAtom())
            config.append([
                rdkit.__version__,
                self.version.pystow_path_key if self.version is not None else '',
                dim_tag,
            ],
            )
            for fp_type in fingerprint:
                fp_group = h5file.create_group(
                    simil_group, repr(fp_type), f'Similarity {repr(fp_type)}',
                )
                particle = create_schema(fp_type.length)
                fp_table = h5file.create_table(
                    fp_group, 'fps', particle, 'Similarity FPs',
                    expectedrows=1_300_000, filters=filters,
                )
                fp_table.attrs.fp_type = fp_type.name
                fp_table.attrs.fp_id = repr(fp_type)
                fp_table.attrs.length = fp_type.length
                fp_table.attrs.fp_params = json.dumps(fp_type.params)

        if parallel:
            self._parallel_create(njobs, fingerprint, progress, total, resolved_n_shards, pattern_holder_bits)
        else:
            self._single_process_create(fingerprint, progress, total, resolved_n_shards, pattern_holder_bits)

        self.load(self.h5_filename)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, fpsubsim_path: str | Path) -> None:
        """Load an existing FPSubSim2 database file.

        :param fpsubsim_path: path to the ``.h5`` database
        :raises ValueError: if *fpsubsim_path* does not exist
        """
        fpsubsim_path = Path(fpsubsim_path)
        if not fpsubsim_path.is_file():
            raise ValueError(f'File does not exist: {fpsubsim_path!r}')
        self.h5_filename = fpsubsim_path

        with tb.open_file(self.h5_filename) as h5file:
            rdkit_version, version_str, dim_tag = h5file.root.config.read()[0]

        if rdkit.__version__ != rdkit_version:
            warnings.warn(
                f'RDKit version mismatch: library was built with {rdkit_version}, '
                f'current version is {rdkit.__version__}. '
                'Consider regenerating the FPSubSim2 library to avoid unexpected behaviour.',
                stacklevel=2,
            )

        self.is3d = (dim_tag == '3D')
        # Reconstruct the PapyrusVersion from the stored old-format string.
        # If the string is empty (database built from a custom SD file), leave as None.
        self.version = PapyrusVersion(version=version_str) if version_str else None
        # Invalidate the fingerprint cache so the next access re-reads the file.
        self._avail_fp: dict | None = None

    # ------------------------------------------------------------------
    # Internal: single-process creation
    # ------------------------------------------------------------------

    def _single_process_create(
            self,
            fingerprint: list[Fingerprint],
            progress: bool,
            total: int | None,
            n_shards: int,
            pattern_holder_bits: int,
    ) -> None:
        """Populate similarity and substructure tables from a single process."""
        if self.h5_filename is None:
            raise RuntimeError('h5_filename not set before _single_process_create()')
        with tb.open_file(self.h5_filename, mode='r+') as h5file:
            libs = [
                SubstructLibrary(CachedSmilesMolHolder(), PatternHolder(pattern_holder_bits))
                for _ in range(n_shards)
            ]
            # shard_mol_ids[s][k] = true idnumber of shard s's k-th AddMol().
            shard_mol_ids: list[list[int]] = [[] for _ in range(n_shards)]
            mappings_table = h5file.root.mol_mappings
            # Map each fingerprint repr → HDF5 table path (consistent key everywhere).
            table_paths = {repr(fp): _fp_table_path(fp) for fp in fingerprint}
            fps = defaultdict(list)
            mappings = []

            with MolSupplier(source=self.sd_file, total=total,
                             show_progress=progress, start_id=1,
                             desc='🔍 Building FPSubSim2 database',
                             ) as supplier:
                for mol_id, rdmol in supplier:
                    shard_idx = _shard_for_mol_id(mol_id, n_shards)
                    libs[shard_idx].AddMol(rdmol)
                    shard_mol_ids[shard_idx].append(mol_id)
                    connectivity, inchikey = _derive_connectivity(rdmol.GetPropsAsDict(), rdmol)
                    mappings.append((mol_id, connectivity, inchikey))

                    for fp_type in fingerprint:
                        fps[repr(fp_type)].append((mol_id, *fp_type.get(rdmol)))

                    if len(fps[repr(fingerprint[0])]) == BATCH_WRITE_SIZE:
                        for fp_type in fingerprint:
                            h5file.get_node(table_paths[repr(fp_type)]).append(
                                fps[repr(fp_type)],
                            )
                        mappings_table.append(mappings)
                        fps, mappings = defaultdict(list), []

                # Flush the final partial batch.
                if fps[repr(fingerprint[0])]:
                    for fp_type in fingerprint:
                        node = h5file.get_node(table_paths[repr(fp_type)])
                        node.append(fps[repr(fp_type)])
                        node.flush()
                    mappings_table.append(mappings)
                    mappings_table.flush()

            for fp_type in fingerprint:
                h5file.get_node(table_paths[repr(fp_type)]).cols.popcnt.create_index(kind='full')
            h5file.root.mol_mappings.cols.idnumber.create_index(kind='full')

            for shard_idx, lib in enumerate(libs):
                shard_table = h5file.get_node(f'/substructure_info/shard_{shard_idx}/substruct_lib')
                lib_bytes = _serialize_substruct_lib(lib)
                shard_table.attrs.padding = _padding_for(lib_bytes)
                lib_bytes = _pad_to_int64(lib_bytes)
                shard_table.append(np.frombuffer(lib_bytes, dtype=np.int64))
                h5file.get_node(f'/substructure_info/shard_{shard_idx}/mol_ids').append(
                    np.array(shard_mol_ids[shard_idx], dtype=np.int64),
                )

        sort_db_file(self.h5_filename, verbose=progress)

    # ------------------------------------------------------------------
    # Internal: parallel creation
    # ------------------------------------------------------------------

    def _parallel_create(
            self,
            njobs: int,
            fingerprint: list[Fingerprint],
            progress: bool,
            total: int | None,
            n_shards: int,
            pattern_holder_bits: int,
    ) -> None:
        """Populate similarity and substructure tables using multiple processes.

        Topology: each worker owns one substructure shard (built in-process)
        and pushes batched rows to ``fp_output_queue``. ``fp_writer`` writes
        the similarity/mapping tables; ``subst_writer`` writes the
        substructure shards, but only after ``fp_writer`` has closed the file
        (``fp_writer_done`` - sequencing, since concurrent HDF5 writes
        aren't safe).

        For a Parquet *sd_file* there is no reader process: each worker reads
        the file directly and parses only its own shard's rows (see
        ``_parquet_shard_worker_process``), so parsing isn't bottlenecked on
        one serial reader. Other formats still use the reader + per-shard
        input queues, since row boundaries need sequential parsing to find.

        The progress bar is owned by this (main) process and updated from
        counts ``fp_writer`` reports over ``progress_queue``.
        """
        if self.h5_filename is None:
            raise RuntimeError('h5_filename not set before _parallel_create()')
        # Pass only (type, params) pairs to worker processes — instances are
        # not safe to share across process boundaries.
        fp_specs = [(type(fp), fp.params) for fp in fingerprint]
        table_paths = {repr(fp): _fp_table_path(fp) for fp in fingerprint}

        n_workers = n_shards  # one shard per worker, built in-process (see docstring)
        use_parquet_workers = str(self.sd_file).endswith('.parquet')

        input_queues: list[multiprocessing.Queue] = []
        reader: multiprocessing.Process | None = None
        if not use_parquet_workers:
            # Bounded so put() blocks naturally - Queue.qsize() isn't portable.
            input_queues = [
                multiprocessing.Queue(maxsize=INPUT_QUEUE_MAXSIZE_BATCHES) for _ in range(n_shards)
            ]
            reader = multiprocessing.Process(
                target=_reader_process,
                args=(self.sd_file, n_shards, total, False, input_queues),
            )

        fp_output_queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=OUTPUT_QUEUE_MAXSIZE_BATCHES_PER_WORKER * n_workers,
        )
        shard_result_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=n_shards)
        # Unbounded: holds only small ints; must not block the writer.
        progress_queue: multiprocessing.Queue | None = multiprocessing.Queue() if progress else None
        fp_writer_done = multiprocessing.Event()

        fp_writer = multiprocessing.Process(
            target=_fp_writer_process,
            args=(self.h5_filename, fp_output_queue, table_paths, progress_queue, fp_writer_done),
        )
        subst_writer = multiprocessing.Process(
            target=_subst_writer_process,
            args=(self.h5_filename, shard_result_queue, n_shards, fp_writer_done),
        )
        if use_parquet_workers:
            workers = [
                multiprocessing.Process(
                    target=_parquet_shard_worker_process,
                    args=(
                        shard_idx, fp_specs, pattern_holder_bits,
                        self.sd_file, n_shards, fp_output_queue, shard_result_queue,
                    ),
                )
                for shard_idx in range(n_workers)
            ]
        else:
            workers = [
                multiprocessing.Process(
                    target=_worker_process,
                    args=(
                        shard_idx, fp_specs, pattern_holder_bits,
                        input_queues[shard_idx], fp_output_queue, shard_result_queue,
                    ),
                )
                for shard_idx in range(n_workers)
            ]

        procs = [p for p in (reader, fp_writer, subst_writer, *workers) if p is not None]
        wait_first = [p for p in (reader, *workers) if p is not None]
        pbar = tqdm(total=total, smoothing=0.0, desc='🔍 Building FPSubSim2 database') if progress else None
        try:
            if reader is not None:
                reader.start()
            fp_writer.start()
            subst_writer.start()
            for w in workers:
                w.start()

            if progress_queue is not None:
                # Poll instead of blocking join() so the bar updates live.
                remaining = list(wait_first)
                while remaining:
                    _pump_progress_queue(progress_queue, pbar)
                    remaining = [p for p in remaining if p.is_alive()]
            for p in wait_first:
                p.join()

            fp_output_queue.put('STOP')

            if progress_queue is not None:
                while fp_writer.is_alive():
                    _pump_progress_queue(progress_queue, pbar)
            fp_writer.join()
            if progress_queue is not None:
                _pump_progress_queue(progress_queue, pbar, timeout=0.0)  # catch any late-arriving count
            subst_writer.join()
        finally:
            # No-op on the happy path; on exception, terminates whatever's
            # still alive (including a subst_writer stuck in .wait()).
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
            for q in [*input_queues, fp_output_queue, shard_result_queue]:
                q.close()
                q.join_thread()
            if progress_queue is not None:
                progress_queue.close()
                progress_queue.join_thread()
            if pbar is not None:
                pbar.close()

        sort_db_file(self.h5_filename, verbose=progress)

    # ------------------------------------------------------------------
    # Fingerprint and substructure library access
    # ------------------------------------------------------------------

    @property
    def available_fingerprints(self) -> dict:
        """Map each stored fingerprint's signature to its :class:`~fingerprint.Fingerprint` instance.

        The result is cached after the first read. The cache is invalidated
        automatically after :meth:`load` or :meth:`add_fingerprint`.
        """
        if self._avail_fp is not None:
            return self._avail_fp
        self._avail_fp = {}
        with tb.open_file(self.h5_filename, mode='r') as h5file:
            for group in h5file.walk_groups('/similarity_info/'):
                if not group._v_name:
                    continue
                fp_table = h5file.get_node(group, 'fps', classname='Table')
                self._avail_fp[fp_table.attrs.fp_id] = get_fp_from_name(
                    fp_table.attrs.fp_type,
                    **json.loads(fp_table.attrs.fp_params),
                )
        return self._avail_fp

    def get_substructure_lib(
            self,
            max_resident_shards: int | None = None,
    ) -> PapyrusSubstructureLibrary | PapyrusShardedSubstructureLibrary:
        """Return the deserialized RDKit substructure library.

        Legacy single-blob databases (``format_version`` absent or ``1``)
        return a :class:`PapyrusSubstructureLibrary`; sharded databases
        (``format_version == 2``) return a
        :class:`PapyrusShardedSubstructureLibrary`.

        :param max_resident_shards: for sharded databases, keep at most this
            many shard libraries resident, loading/discarding the rest per
            query; ``None`` (default) keeps every shard resident. Ignored
            for legacy databases.
        :raises ValueError: if the database file does not exist yet
        """
        if self.h5_filename is None or not self.h5_filename.is_file():
            raise ValueError('Database file must be created first.')
        with tb.open_file(self.h5_filename, mode='r') as h5file:
            subst_group = h5file.root.substructure_info
            format_version = getattr(subst_group._v_attrs, 'format_version', 1)
            if format_version == 1:
                padding = subst_group.substruct_lib.attrs.padding
                data = subst_group.substruct_lib.read()
                raw = data.tobytes('C')
                raw = raw[:-padding] if padding else raw
                with BytesIO(raw) as stream:
                    lib = PapyrusSubstructureLibrary(self.h5_filename)
                    lib.InitFromStream(stream)
                return lib

            n_shards = subst_group._v_attrs.n_shards
            shard_data = []
            shard_mol_ids = []
            for shard_idx in range(n_shards):
                shard_node = h5file.get_node(f'/substructure_info/shard_{shard_idx}/substruct_lib')
                padding = shard_node.attrs.padding
                data = shard_node.read()
                raw = data.tobytes('C')
                shard_data.append(raw[:-padding] if padding else raw)
                mol_ids_node = h5file.get_node(f'/substructure_info/shard_{shard_idx}/mol_ids')
                shard_mol_ids.append(mol_ids_node.read().tolist())

        return PapyrusShardedSubstructureLibrary(
            self.h5_filename, shard_data, shard_mol_ids, max_resident_shards=max_resident_shards,
        )

    def get_similarity_lib(
            self,
            fp_signature: str | None = None,
            cuda: bool | Literal['auto'] = False,
            in_memory_fps: bool = True,
            fps_sort: bool = False,
    ) -> FPSubSim2Engine | FPSubSim2CudaEngine:
        """Return a similarity search engine for the requested fingerprint.

        The CPU engine (:class:`FPSubSim2Engine`) is the default, requiring
        only ``tables`` + ``FPSim2`` (no ``cupy``/CUDA). For RAM-constrained
        use, pass ``in_memory_fps=False`` and call the returned engine's
        ``on_disk_similarity``/``on_disk_tversky`` instead of
        ``similarity``/``tversky``.

        :param fp_signature: fingerprint signature; defaults to the first
            available one when ``None``
        :param cuda: ``True`` for the GPU engine (raises :exc:`ImportError`
            if unavailable), ``False`` (default) for CPU, or ``'auto'`` to
            try GPU and fall back to CPU with a warning. ``in_memory_fps``/
            ``fps_sort`` are ignored by the CUDA engine.
        :param in_memory_fps: load the whole fingerprint matrix into RAM
            (CPU engine only); ``False`` uses the on-disk/chunked methods
        :param fps_sort: sort fingerprints by popcount when loading
            in-memory (CPU engine only); unnecessary for databases already
            sorted via :func:`sort_db_file`
        :raises ValueError: if the database does not exist or the signature is
            not found in the database
        """
        if self.h5_filename is None or not self.h5_filename.is_file():
            raise ValueError('Database file must be created first.')
        available = self.available_fingerprints
        if fp_signature is None:
            fp_signature = next(iter(available))
        if fp_signature not in available:
            raise ValueError(
                f'Fingerprint {fp_signature!r} not available. '
                f'Choose one of: {list(available)}',
            )
        if cuda == 'auto':
            try:
                return FPSubSim2CudaEngine(self.h5_filename, fp_signature)
            except Exception as exc:  # any GPU init failure should fall back, not just ImportError
                warnings.warn(
                    f'GPU-accelerated search unavailable ({exc!r}); falling back to the CPU engine.',
                    stacklevel=2,
                )
                cuda = False
        if cuda:
            return FPSubSim2CudaEngine(self.h5_filename, fp_signature)
        return FPSubSim2Engine(
            self.h5_filename, fp_signature,
            in_memory_fps=in_memory_fps, fps_sort=fps_sort,
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_fingerprint(
            self,
            fingerprint: Fingerprint,
            papyrus_sd_file: str,
            progress: bool = True,
            total: int | None = None,
    ) -> None:
        """Add a new fingerprint type to the database.

        :param fingerprint: the :class:`~fingerprint.Fingerprint` to add
        :param papyrus_sd_file: SD file containing the structures
        :param progress: display a progress bar
        :param total: molecule count for the progress bar
        """
        if self.h5_filename is None:
            raise RuntimeError('call create_from_papyrus()/create()/load() before add_fingerprint()')
        signature = repr(fingerprint)
        available_fps = list(self.available_fingerprints.keys())
        if signature in available_fps:
            print(f'Fingerprint {signature!r} is already available.')
            return
        backend = PyTablesMultiFpStorageBackend(self.h5_filename, available_fps[0])
        backend.change_fp_for_append(fingerprint)
        backend.append_fps(
            MolSupplier(source=papyrus_sd_file),
            total=total,
            progress=progress,
        )
        # Invalidate the fingerprint cache.
        self._avail_fp = None

    def add_molecules(
            self,
            papyrus_sd_file: str,
            progress: bool = True,
            total: int | None = None,
    ) -> None:
        """Append new molecules to the mapping table, all fingerprint tables, and the substructure library.

        :param papyrus_sd_file: SD file containing new structures to add
        :param progress: display a progress bar
        :param total: molecule count for the progress bar
        """
        if self.h5_filename is None:
            raise RuntimeError('call create_from_papyrus()/create()/load() before add_molecules()')

        with tb.open_file(self.h5_filename, mode='r') as h5file:
            # +1: continue after the last idnumber, don't reuse it.
            start_id = max(
                (row['idnumber'] for row in h5file.root.mol_mappings.iterrows()), default=0,
            ) + 1
            format_version = getattr(h5file.root.substructure_info._v_attrs, 'format_version', 1)

        mappings = []
        with MolSupplier(source=papyrus_sd_file, total=total,
                         show_progress=progress, start_id=start_id,
                         ) as supplier:
            for mol_id, rdmol in supplier:
                connectivity, inchikey = _derive_connectivity(rdmol.GetPropsAsDict(), rdmol)
                mappings.append((mol_id, connectivity, inchikey))
        with tb.open_file(self.h5_filename, mode='a') as h5file:
            h5file.root.mol_mappings.append(mappings)
            h5file.root.mol_mappings.flush()
            h5file.root.mol_mappings.cols.idnumber.reindex()

        for signature, _fingerprint in self.available_fingerprints.items():
            backend = PyTablesMultiFpStorageBackend(self.h5_filename, signature)
            backend.append_fps(
                MolSupplier(source=papyrus_sd_file),
                total=total,
                progress=progress,
                sort=False,
            )

        if format_version == 1:
            substruct_lib = cast(PapyrusSubstructureLibrary, self.get_substructure_lib())
            with MolSupplier(source=papyrus_sd_file, total=total,
                             show_progress=progress, start_id=start_id,
                             ) as supplier:
                for _, rdmol in supplier:
                    substruct_lib.AddMol(rdmol)

            lib_bytes = _serialize_substruct_lib(substruct_lib)
            padding = _padding_for(lib_bytes)
            lib_bytes = _pad_to_int64(lib_bytes)

            with tb.open_file(self.h5_filename, mode='a') as h5file:
                h5file.remove_node(h5file.root.substructure_info.substruct_lib)
                arr = h5file.create_earray(
                    h5file.root.substructure_info, 'substruct_lib',
                    tb.UInt64Atom(), (0,), 'Substructure search library',
                )
                arr.attrs.padding = padding
                arr.append(np.frombuffer(lib_bytes, dtype=np.int64))
        else:
            sharded_lib = cast(PapyrusShardedSubstructureLibrary, self.get_substructure_lib())
            new_mol_ids_by_shard: defaultdict[int, list[int]] = defaultdict(list)
            with MolSupplier(source=papyrus_sd_file, total=total,
                             show_progress=progress, start_id=start_id,
                             ) as supplier:
                for mol_id, rdmol in supplier:
                    shard_idx = _shard_for_mol_id(mol_id, sharded_lib.n_shards)
                    lib = sharded_lib._resident[shard_idx]
                    if lib is None:
                        lib = sharded_lib._load_shard(shard_idx)
                        sharded_lib._resident[shard_idx] = lib
                    lib.AddMol(rdmol)
                    new_mol_ids_by_shard[shard_idx].append(mol_id)

            # Only shards that actually received new molecules are rewritten.
            with tb.open_file(self.h5_filename, mode='a') as h5file:
                for shard_idx, new_ids in new_mol_ids_by_shard.items():
                    # Non-None: populated for every touched shard in the loop above.
                    lib = cast(PapyrusSubstructureLibrary, sharded_lib._resident[shard_idx])
                    lib_bytes = _serialize_substruct_lib(lib)
                    padding = _padding_for(lib_bytes)
                    lib_bytes = _pad_to_int64(lib_bytes)

                    h5file.remove_node(f'/substructure_info/shard_{shard_idx}/substruct_lib')
                    arr = h5file.create_earray(
                        h5file.get_node(f'/substructure_info/shard_{shard_idx}'), 'substruct_lib',
                        tb.UInt64Atom(), (0,), f'Substructure search library shard {shard_idx}',
                    )
                    arr.attrs.padding = padding
                    arr.append(np.frombuffer(lib_bytes, dtype=np.int64))

                    h5file.get_node(f'/substructure_info/shard_{shard_idx}/mol_ids').append(
                        np.array(new_ids, dtype=np.int64),
                    )

        sort_db_file(self.h5_filename, verbose=progress)


# ---------------------------------------------------------------------------
# Substructure-library serialisation helpers
# ---------------------------------------------------------------------------

def _padding_for(data: bytes) -> int:
    """Return the number of zero bytes needed to align *data* to 8 bytes."""
    remainder = len(data) % 8
    return (8 - remainder) if remainder else 0


def _serialize_substruct_lib(lib: SubstructLibrary) -> bytes:
    """Serialise *lib* and return the raw bytes (without padding)."""
    return lib.Serialize()


def _pad_to_int64(data: bytes) -> bytes:
    """Zero-pad *data* to a multiple of 8 bytes (``int64`` alignment)."""
    padding = _padding_for(data)
    return data + b'\x00' * padding if padding else data


# ---------------------------------------------------------------------------
# Multi-process worker functions (module-level so they are pickleable)
# ---------------------------------------------------------------------------

def _reader_process(
        sd_file: str,
        n_shards: int,
        total: int | None,
        progress: bool,
        input_queues: list[multiprocessing.Queue],
) -> None:
    """Read molecules from *sd_file* and route batches to per-shard worker queues."""
    pending: list[list[tuple]] = [[] for _ in range(n_shards)]
    with MolSupplier(source=sd_file, total=total, show_progress=progress, start_id=1) as supplier:
        for mol_id, rdmol in supplier:
            shard_idx = _shard_for_mol_id(mol_id, n_shards)
            pending[shard_idx].append((mol_id, rdmol, rdmol.GetPropsAsDict()))
            if len(pending[shard_idx]) >= READER_BATCH_SIZE:
                input_queues[shard_idx].put(pending[shard_idx])
                pending[shard_idx] = []
    for shard_idx, batch in enumerate(pending):
        if batch:
            input_queues[shard_idx].put(batch)
    # Send one termination sentinel per shard queue.
    for q in input_queues:
        q.put('END')


def _worker_process(
        shard_idx: int,
        fp_specs: list[tuple[type, dict]],
        pattern_holder_bits: int,
        input_queue: multiprocessing.Queue,
        fp_output_queue: multiprocessing.Queue,
        shard_result_queue: multiprocessing.Queue,
) -> None:
    """Build one substructure shard in-process and compute fingerprints for its molecules.

    ``AddMol()`` happens locally; only this worker's own serialized shard
    bytes are sent onward (once, at the end) via *shard_result_queue*.
    """
    lib = SubstructLibrary(CachedSmilesMolHolder(), PatternHolder(pattern_holder_bits))
    fpers = [fp_cls(**fp_params) for fp_cls, fp_params in fp_specs]
    mol_ids: list[int] = []  # mol_ids[k] = true idnumber of the k-th AddMol()

    mappings_batch: list[tuple] = []
    fps_batch: defaultdict[str, list] = defaultdict(list)

    def _flush() -> None:
        nonlocal mappings_batch, fps_batch
        if mappings_batch:
            fp_output_queue.put(('rows', mappings_batch, dict(fps_batch)))
        mappings_batch = []
        fps_batch = defaultdict(list)

    while True:
        data = input_queue.get()
        if data == 'END':
            break
        for mol_id, rdmol, props in data:
            lib.AddMol(rdmol)
            mol_ids.append(mol_id)
            connectivity, inchikey = _derive_connectivity(props, rdmol)
            mappings_batch.append((mol_id, connectivity, inchikey))
            for fper in fpers:
                fps_batch[repr(fper)].append((mol_id, *fper.get(rdmol)))
            if len(mappings_batch) >= WORKER_IPC_BATCH_SIZE:
                _flush()
    _flush()

    lib_bytes = _serialize_substruct_lib(lib)
    padding = _padding_for(lib_bytes)
    lib_bytes = _pad_to_int64(lib_bytes)
    shard_result_queue.put((shard_idx, lib_bytes, padding, mol_ids))


def _parquet_shard_worker_process(
        shard_idx: int,
        fp_specs: list[tuple[type, dict]],
        pattern_holder_bits: int,
        parquet_path: str | Path,
        n_shards: int,
        fp_output_queue: multiprocessing.Queue,
        shard_result_queue: multiprocessing.Queue,
) -> None:
    """Build one substructure shard by reading *parquet_path* directly.

    Opens its own ``pq.ParquetFile`` handle and parses only the rows whose
    ``mol_id`` belongs to *shard_idx* - unlike :func:`_worker_process`, no
    reader process feeds it.
    """
    lib = SubstructLibrary(CachedSmilesMolHolder(), PatternHolder(pattern_holder_bits))
    fpers = [fp_cls(**fp_params) for fp_cls, fp_params in fp_specs]
    mol_ids: list[int] = []  # mol_ids[k] = true idnumber of the k-th AddMol()
    failed_ids: list[int] = []

    mappings_batch: list[tuple] = []
    fps_batch: defaultdict[str, list] = defaultdict(list)

    def _flush() -> None:
        nonlocal mappings_batch, fps_batch
        if mappings_batch:
            fp_output_queue.put(('rows', mappings_batch, dict(fps_batch)))
        mappings_batch = []
        fps_batch = defaultdict(list)

    parquet_file = pq.ParquetFile(parquet_path)
    prop_cols = [name for name in parquet_file.schema_arrow.names if name != 'ctab']

    row_id = 1  # start_id, matching MolSupplier's default
    for batch in parquet_file.iter_batches():
        n_rows = batch.num_rows
        shard_positions = [
            i for i in range(n_rows) if _shard_for_mol_id(row_id + i, n_shards) == shard_idx
        ]
        if shard_positions:
            ctab_col = batch.column('ctab')
            prop_col_arrays = {name: batch.column(name) for name in prop_cols}
            for i in shard_positions:
                mol_id = row_id + i
                ctab = ctab_col[i].as_py()
                if ctab is None:
                    failed_ids.append(mol_id)
                    continue
                rdmol = Chem.MolFromMolBlock(ctab)
                if rdmol is None:
                    failed_ids.append(mol_id)
                    continue
                for name in prop_cols:
                    value = prop_col_arrays[name][i].as_py()
                    if value is not None:
                        rdmol.SetProp(name, str(value))

                lib.AddMol(rdmol)
                mol_ids.append(mol_id)
                connectivity, inchikey = _derive_connectivity(rdmol.GetPropsAsDict(), rdmol)
                mappings_batch.append((mol_id, connectivity, inchikey))
                for fper in fpers:
                    fps_batch[repr(fper)].append((mol_id, *fper.get(rdmol)))
                if len(mappings_batch) >= WORKER_IPC_BATCH_SIZE:
                    _flush()
        row_id += n_rows
    _flush()

    if failed_ids:
        preview = ', '.join(map(str, failed_ids[:10]))
        more = f', +{len(failed_ids) - 10} more' if len(failed_ids) > 10 else ''
        warnings.warn(
            f'{len(failed_ids)} molecule(s) could not be parsed and were skipped '
            f'(indices: {preview}{more}).',
            stacklevel=2,
        )

    lib_bytes = _serialize_substruct_lib(lib)
    padding = _padding_for(lib_bytes)
    lib_bytes = _pad_to_int64(lib_bytes)
    shard_result_queue.put((shard_idx, lib_bytes, padding, mol_ids))


def _fp_writer_process(
        h5_filename: str | Path,
        fp_output_queue: multiprocessing.Queue,
        table_paths: dict,
        progress_queue: multiprocessing.Queue | None,
        fp_writer_done: multiprocessing.synchronize.Event,
) -> None:
    """Consume fingerprint/mapping row batches from workers and write them to the HDF5 file.

    Sets *fp_writer_done* in a ``finally`` so an exception still releases
    the waiting ``_subst_writer_process``.

    Reports progress by count over *progress_queue* rather than owning a
    ``tqdm`` bar (unsafe in a forked child - see ``_parallel_create``).
    """
    mappings_insert: list[tuple] = []
    similarity_insert: defaultdict[str, list] = defaultdict(list)

    try:
        with tb.open_file(h5_filename, mode='r+') as h5file:
            while True:
                data = fp_output_queue.get()
                if data == 'STOP':
                    # Flush remaining data.
                    if mappings_insert:
                        h5file.root.mol_mappings.append(mappings_insert)
                    for fp_id, fp_rows in similarity_insert.items():
                        if fp_rows:
                            h5file.get_node(table_paths[fp_id]).append(fp_rows)
                    # Create indices.
                    for path in table_paths.values():
                        h5file.get_node(path).cols.popcnt.create_index(kind='full')
                    h5file.root.mol_mappings.cols.idnumber.create_index(kind='full')
                    break

                _, mappings_batch, fps_batch = data
                mappings_insert.extend(mappings_batch)
                if progress_queue is not None:
                    progress_queue.put(len(mappings_batch))
                for fp_id, fp_rows in fps_batch.items():
                    similarity_insert[fp_id].extend(fp_rows)

                if len(mappings_insert) > BATCH_WRITE_SIZE:
                    h5file.root.mol_mappings.append(mappings_insert)
                    h5file.root.mol_mappings.flush()
                    mappings_insert = []

                if any(len(v) > BATCH_WRITE_SIZE for v in similarity_insert.values()):
                    for fp_id, fp_rows in similarity_insert.items():
                        node = h5file.get_node(table_paths[fp_id])
                        node.append(fp_rows)
                        node.flush()
                    similarity_insert = defaultdict(list)

        with tb.open_file(h5_filename, mode='r+') as h5file:
            h5file.root.mol_mappings.cols.idnumber.reindex()
    finally:
        fp_writer_done.set()


def _subst_writer_process(
        h5_filename: str | Path,
        shard_result_queue: multiprocessing.Queue,
        n_shards: int,
        fp_writer_done: multiprocessing.synchronize.Event,
) -> None:
    """Collect each worker's serialized substructure shard and write it to the HDF5 file.

    Waits for *fp_writer_done* first, since concurrent HDF5 writes aren't safe.
    """
    shards: dict[int, tuple[bytes, int, list[int]]] = {}
    while len(shards) < n_shards:
        shard_idx, lib_bytes, padding, mol_ids = shard_result_queue.get()
        shards[shard_idx] = (lib_bytes, padding, mol_ids)

    fp_writer_done.wait()

    with tb.open_file(h5_filename, mode='r+') as h5file:
        for shard_idx, (lib_bytes, padding, mol_ids) in shards.items():
            shard_table = h5file.get_node(f'/substructure_info/shard_{shard_idx}/substruct_lib')
            shard_table.attrs.padding = padding
            shard_table.append(np.frombuffer(lib_bytes, dtype=np.int64))
            h5file.get_node(f'/substructure_info/shard_{shard_idx}/mol_ids').append(
                np.array(mol_ids, dtype=np.int64),
            )


# ---------------------------------------------------------------------------
# PyTables storage backend
# ---------------------------------------------------------------------------

class PyTablesMultiFpStorageBackend(BaseStorageBackend):
    """PyTables-backed storage backend for multi-fingerprint FPSubSim2 databases."""

    def __init__(
            self,
            fp_filename: str | Path,
            fp_signature: str,
            in_memory_fps: bool = True,
            fps_sort: bool = False,
    ) -> None:
        """Open *fp_filename* and select the *fp_signature* fingerprint table to read/append to."""
        _check_optional_deps()
        # BaseStorageBackend defines no __init__ of its own (pure ABC).
        self.fp_filename = fp_filename
        self.name = 'pytables'

        with tb.open_file(self.fp_filename, mode='r') as fp_file:
            self._fp_table_mappings: dict[str, list[str]] = {}
            for group in fp_file.walk_groups('/similarity_info/'):
                if not group._v_name:
                    continue
                fp_table = fp_file.get_node(group, 'fps', classname='Table')
                base = f'/similarity_info/{group._v_name}'
                self._fp_table_mappings[fp_table.attrs.fp_id] = [
                    f'{base}/fps',
                    f'{base}/popcounts',
                ]

        if fp_signature not in self._fp_table_mappings:
            raise ValueError(
                f'Fingerprint {fp_signature!r} not available. '
                f'Choose one of: {", ".join(self._fp_table_mappings)}',
            )

        self._current_fp = fp_signature
        self._current_fp_path = self._fp_table_mappings[fp_signature][0]
        self._current_popcounts_path = self._fp_table_mappings[fp_signature][1]
        self.fp_type, self.fp_params, self.rdkit_ver = self.read_parameters()
        self._fp_func = get_fp_from_name(self.fp_type, **self.fp_params)

        if in_memory_fps:
            self.load_fps(in_memory_fps, fps_sort)
        self.load_popcnt_bins(fps_sort)

        with tb.open_file(self.fp_filename, mode='r') as fp_file:
            self.chunk_size = fp_file.get_node(self._current_fp_path).chunkshape[0] * 120

    def read_parameters(self) -> tuple[str, dict, str]:
        """Read fingerprint metadata for the currently selected fingerprint."""
        with tb.open_file(self.fp_filename, mode='r') as fp_file:
            rdkit_ver = fp_file.root.config[0]
            fp_table = fp_file.get_node(self._current_fp_path)
            fp_type = fp_table.attrs.fp_type
            fp_params = json.loads(fp_table.attrs.fp_params)
        return fp_type, fp_params, rdkit_ver

    def get_fps_chunk(self, chunk_range: tuple[int, int]) -> np.ndarray:
        """Read fingerprints in row range *chunk_range* (start, stop) from the current table."""
        with tb.open_file(self.fp_filename, mode='r') as fp_file:
            return fp_file.get_node(self._current_fp_path)[slice(*chunk_range)]

    def load_popcnt_bins(self, fps_sort: bool) -> None:
        """Load popcount bins, computing them from the fingerprints if *fps_sort*."""
        if fps_sort:
            self.popcnt_bins = self.calc_popcnt_bins(self.fps)
        else:
            with tb.open_file(self.fp_filename, mode='r') as fp_file:
                self.popcnt_bins = fp_file.get_node(self._current_popcounts_path).read()

    def load_fps(self, in_memory_fps: bool, fps_sort: bool) -> None:
        """Load fingerprints into memory for the currently selected fingerprint type."""
        with tb.open_file(self.fp_filename, mode='r') as fp_file:
            fps = fp_file.get_node(self._current_fp_path)[:]
            if fps_sort:
                fps.sort(order='popcnt')
        num_fields = len(fps[0])
        fps = fps.view('<u8')
        fps = fps.reshape(fps.size // num_fields, num_fields)
        self.fps = fps

    def delete_fps(self, ids_list: list[int]) -> None:
        """Delete fingerprint rows by their integer IDs."""
        with tb.open_file(self.fp_filename, mode='a') as fp_file:
            fps_table = fp_file.get_node(self._current_fp_path)
            for fp_id in ids_list:
                to_delete = [
                    row.nrow
                    for row in fps_table.where(f'fp_id == {fp_id}')
                ]
                if to_delete:
                    fps_table.remove_row(to_delete[0])

    def append_fps(
            self,
            supplier: MolSupplier,
            progress: bool = True,
            total: int | None = None,
            sort: bool = True,
    ) -> None:
        """Append fingerprints for molecules from *supplier* to the database."""
        with tb.open_file(self.fp_filename, mode='a') as fp_file:
            fps_table = fp_file.get_node(self._current_fp_path)
            # +1: continue after the last existing id, don't reuse it.
            start_id = max(
                (row['fp_id'] for row in fps_table.iterrows()), default=0,
            ) + 1
            supplier.set_start_progress_total(start_id, progress, total)
            fps = []
            for mol_id, rdmol in supplier:
                fps.append((mol_id, *self._fp_func.get(rdmol)))
                if len(fps) == BATCH_WRITE_SIZE:
                    fps_table.append(fps)
                    fps = []
            if fps:
                fps_table.append(fps)
        if sort:
            sort_db_file(self.fp_filename, verbose=progress)

    def change_fp_for_append(self, fingerprint: Fingerprint) -> None:
        """Create an empty table for *fingerprint* and select it for appending.

        :param fingerprint: the new :class:`~fingerprint.Fingerprint` type to add
        """
        self._current_fp = repr(fingerprint)
        particle = create_schema(fingerprint.length)
        filters = tb.Filters()

        with tb.open_file(self.fp_filename, mode='a') as fp_file:
            fp_group = fp_file.create_group(
                '/similarity_info/', self._current_fp,
                f'Similarity {self._current_fp}',
            )
            fp_table = fp_file.create_table(
                fp_group, 'fps', particle, 'Similarity FPs',
                expectedrows=1_300_000, filters=filters,
            )
            fp_table.attrs.fp_type = fingerprint.name
            fp_table.attrs.fp_id = self._current_fp
            fp_table.attrs.length = fingerprint.length
            fp_table.attrs.fp_params = json.dumps(fingerprint.params)
            fp_file.create_vlarray(
                fp_group, 'popcounts', tb.ObjectAtom(),
                f'Popcounts of {fp_group._v_name}',
            )

        self._current_fp_path = f'/similarity_info/{fp_group._v_name}/fps'
        self._current_popcounts_path = f'/similarity_info/{fp_group._v_name}/popcounts'
        self.fp_type, self.fp_params, self.rdkit_ver = self.read_parameters()
        self._fp_func = get_fp_from_name(self.fp_type, **self.fp_params)
        print(f'Empty table created for {self._current_fp!r}. '
              'Call `append_fps` to populate it.',
              )


# ---------------------------------------------------------------------------
# Mapping helper mixin
# ---------------------------------------------------------------------------

class _MappingMixin:
    """Mixin that provides molecule-ID → Papyrus-identifier lookup."""

    fp_filename: str | Path  # supplied by concrete subclass

    def _get_mapping(self, ids: list[int] | int) -> pl.DataFrame:
        """Return a DataFrame with Papyrus identifiers for the given integer *ids*.

        Reads the table once and joins in memory - a per-id ``where()`` loop
        doesn't scale to large hit counts.

        :param ids: one or more molecule IDs from the similarity/substructure result
        :raises ValueError: if any ID is not an integer or is not in the database
        """
        if not isinstance(ids, list):
            ids = [ids]
        if not ids:
            raise ValueError('At least one index must be supplied.')
        for i in ids:
            if int(i) != i:
                raise ValueError(f'All indices must be integers; got {i!r}.')

        with tb.open_file(self.fp_filename) as fp_file:
            mappings_table = fp_file.root.mol_mappings
            colnames = mappings_table.cols._v_colnames
            data = mappings_table.read()

        full = _decode_bytes_df(pl.DataFrame({c: data[c] for c in colnames}))
        result = pl.DataFrame({'idnumber': pl.Series(ids, dtype=pl.Int64)}).join(
            full, on='idnumber', how='left',
        )
        missing = result.filter(pl.col('InChIKey').is_null())
        if missing.height:
            raise ValueError(f'Index {missing["idnumber"][0]} not found in the database.')
        return result


# ---------------------------------------------------------------------------
# Similarity search engines
# ---------------------------------------------------------------------------

class BaseMultiFpEngine(_MappingMixin, BaseEngine, ABC):
    """Abstract base for CPU and GPU similarity search engines."""

    def __init__(
            self,
            fp_filename: str | Path,
            fp_signature: str,
            storage_backend: str,
            in_memory_fps: bool,
            fps_sort: bool,
    ) -> None:
        """Open the fingerprint database and configure the storage backend."""
        _check_optional_deps()
        self.fp_filename = fp_filename
        self.in_memory_fps = in_memory_fps
        if storage_backend == 'pytables':
            self.storage = PyTablesMultiFpStorageBackend(
                fp_filename, fp_signature,
                in_memory_fps=in_memory_fps,
                fps_sort=fps_sort,
            )

    def load_query(self, query_string: str, full_sanitization: bool = True) -> np.ndarray:
        """Parse a SMILES, InChI, or molblock query string into a fingerprint array.

        :param query_string: SMILES, InChI, or molblock
        :param full_sanitization: forwarded to RDKit's molecule loader; must
            be accepted since FPSim2's own search methods call this with it
        :raises ValueError: if the molecule cannot be parsed
        """
        rdmol = load_molecule(query_string, full_sanitization=full_sanitization)
        if rdmol is None:
            raise ValueError(f'Could not parse query molecule: {query_string!r}')
        fp = get_fp_from_name(self.storage.fp_type, **self.storage.fp_params).get(rdmol)
        return np.array((0, *fp), dtype=np.uint64)


class FPSubSim2Engine(BaseMultiFpEngine, FPSim2Engine):
    """CPU-based similarity search engine for FPSubSim2 databases.

    The default engine (see :meth:`FPSubSim2.get_similarity_lib`); needs
    only ``tables`` + ``FPSim2``, no ``cupy``/CUDA. With
    ``in_memory_fps=False``, use ``on_disk_similarity``/``on_disk_tversky``
    instead of ``similarity``/``tversky``.
    """

    def __init__(
            self,
            fp_filename: str | Path,
            fp_signature: str,
            in_memory_fps: bool = True,
            fps_sort: bool = False,
            storage_backend: str = 'pytables',
    ) -> None:
        """Open an FPSubSim2 database for CPU-based similarity search."""
        super().__init__(
            fp_filename=fp_filename,
            fp_signature=fp_signature,
            storage_backend=storage_backend,
            in_memory_fps=in_memory_fps,
            fps_sort=fps_sort,
        )
        self.empty_sim = np.ndarray((0,), dtype=[('mol_id', '<u4'), ('coeff', '<f4')])
        self.empty_subs = np.ndarray((0,), dtype='<u4')

    def _score_col(self, metric: str, threshold: float) -> str:
        return f'{metric} > {threshold} ({self.storage._current_fp})'

    def similarity(self, query_string: str, threshold: float, n_workers: int = 1) -> pl.DataFrame:
        """In-memory Tanimoto similarity search."""
        # By keyword: FPSim2Engine's signature has gained positional params
        # ahead of n_workers across versions.
        raw = FPSim2Engine.similarity(self, query_string, threshold, n_workers=n_workers)
        return _build_result_df(raw, self._get_mapping, self._score_col('Tanimoto', threshold))

    def on_disk_similarity(
            self, query_string: str, threshold: float,
            n_workers: int = 1, chunk_size: int = 0,
    ) -> pl.DataFrame:
        """On-disk Tanimoto similarity search."""
        raw = FPSim2Engine.on_disk_similarity(
            self, query_string, threshold, n_workers=n_workers, chunk_size=chunk_size,
        )
        return _build_result_df(raw, self._get_mapping, self._score_col('Tanimoto', threshold))

    def tversky(
            self, query_string: str, threshold: float,
            a: float, b: float, n_workers: int = 1,
    ) -> pl.DataFrame:
        """In-memory Tversky similarity search."""
        raw = FPSim2Engine.tversky(self, query_string, threshold, a, b, n_workers=n_workers)
        return _build_result_df(raw, self._get_mapping, self._score_col('Tversky', threshold))

    def on_disk_tversky(
            self, query_string: str, threshold: float,
            a: float, b: float,
            n_workers: int = 1, chunk_size: int | None = None,
    ) -> pl.DataFrame:
        """On-disk Tversky similarity search."""
        raw = FPSim2Engine.on_disk_tversky(
            self, query_string, threshold, a, b, n_workers=n_workers, chunk_size=chunk_size,
        )
        return _build_result_df(raw, self._get_mapping, self._score_col('Tversky', threshold))

    def substructure(self, query_string: str, n_workers: int = 1):
        """Not supported here; use the FPSubSim2 substructure library instead."""
        raise NotImplementedError(
            'Use the FPSubSim2 substructure library (get_substructure_lib) '
            'for exact subgraph isomorphism.',
        )

    def on_disk_substructure(self, query_string: str, n_workers: int = 1, chunk_size: int | None = None):
        """Not supported here; use the FPSubSim2 substructure library instead."""
        raise NotImplementedError(
            'Use the FPSubSim2 substructure library (get_substructure_lib) '
            'for exact subgraph isomorphism.',
        )


class FPSubSim2CudaEngine(BaseMultiFpEngine, FPSim2CudaEngine):
    """GPU-accelerated similarity search engine for FPSubSim2 databases.

    Requires ``cupy`` and a CUDA-capable GPU; raises :exc:`ImportError`
    otherwise. Always loads the fingerprint matrix fully into GPU memory -
    ``in_memory_fps``/``fps_sort`` aren't configurable here. Use
    :class:`FPSubSim2Engine` for CPU-only or RAM-light search.
    """

    def __init__(
            self,
            fp_filename: str | Path,
            fp_signature: str,
            storage_backend: str = 'pytables',
            kernel: str = 'raw',
    ) -> None:
        """Open an FPSubSim2 database for GPU-accelerated similarity search."""
        if not HAS_CUPY:
            raise ImportError('cupy is required for GPU-accelerated search.')
        if kernel not in ('raw', 'element_wise'):
            raise ValueError("kernel must be 'raw' or 'element_wise'.")
        if kernel == 'element_wise' and not hasattr(FPSim2CudaEngine, 'ew_kernel'):
            raise NotImplementedError(
                "The installed FPSim2 version's CUDA engine no longer "
                "provides an element-wise kernel (only 'raw'). "
                "Use kernel='raw' instead.",
            )
        super().__init__(
            fp_filename=fp_filename,
            fp_signature=fp_signature,
            storage_backend=storage_backend,
            in_memory_fps=True,
            fps_sort=False,
        )
        self.kernel = kernel
        cp = _ensure_cupy()
        # self.fps is set by FPSim2CudaEngine's own real __init__ - unreachable
        # without cupy+FPSim2 installed.
        if kernel == 'raw':  # pragma: no cover
            self.cuda_db = cp.asarray(self.fps[:, 1:-1])
            self.cuda_ids = cp.asarray(self.fps[:, 0])
            # Name matches FPSim2CudaEngine's own _raw_kernel_search().
            self.cuda_db_popcnts = cp.asarray(self.fps[:, -1])
            self.cupy_kernel = cp.RawKernel(
                self.raw_kernel.format(block=self.cuda_db.shape[1]),
                name='taniRAW',
                options=('-std=c++14',),
            )
        else:  # pragma: no cover
            self.cuda_db = cp.asarray(self.fps)
            self.cupy_kernel = cp.ElementwiseKernel(
                in_params='raw T db, raw U query, uint64 in_width, float32 threshold',
                out_params='raw V out',
                operation=self.ew_kernel,
                name='taniEW',
                options=('-std=c++14',),
                reduce_dims=False,
            )

    def similarity(self, query_string: str, threshold: float) -> pl.DataFrame:  # pragma: no cover
        """GPU Tanimoto similarity search."""
        raw = FPSim2CudaEngine.similarity(self, query_string, threshold)
        return _build_result_df(
            raw, self._get_mapping,
            f'Tanimoto > {threshold} ({self.storage._current_fp})',
        )

    def tversky(self, query_string: str, threshold: float, a: float, b: float) -> pl.DataFrame:  # pragma: no cover
        """GPU Tversky similarity search.

        :raises NotImplementedError: if the installed FPSim2 CUDA engine
            doesn't provide Tversky (only Tanimoto) - use
            :class:`FPSubSim2Engine` (CPU) instead.
        """
        if not hasattr(FPSim2CudaEngine, 'tversky'):
            raise NotImplementedError(
                "The installed FPSim2 version's CUDA engine does not provide "
                'Tversky search (only Tanimoto). Use FPSubSim2Engine (CPU) '
                'instead, via get_similarity_lib(cuda=False).',
            )
        raw = FPSim2CudaEngine.tversky(self, query_string, threshold, a, b)
        return _build_result_df(
            raw, self._get_mapping,
            f'Tversky > {threshold} ({self.storage._current_fp})',
        )


# ---------------------------------------------------------------------------
# Substructure library with Papyrus identifier mapping
# ---------------------------------------------------------------------------

class PapyrusSubstructureLibrary(_MappingMixin, SubstructLibrary):
    """RDKit ``SubstructLibrary`` extended with Papyrus identifier lookup.

    Instances are typically obtained via :meth:`FPSubSim2.get_substructure_lib`
    rather than constructed directly.

    .. note::
        ``GetMatches()`` returns 0-based positions, not ``idnumber`` (1-based,
        can have gaps). When *mol_ids* is supplied, positions are translated
        through it first. Legacy (``format_version == 1``) databases predate
        this mapping, so *mol_ids* is ``None`` there and raw positions are
        used directly - a known limitation for that format.
    """

    def __init__(self, fp_filename: str | Path, mol_ids: list[int] | None = None) -> None:
        """Bind this substructure library to the FPSubSim2 database at *fp_filename*."""
        _check_optional_deps()
        # Placeholder holders - InitFromStream() below reconstructs the real
        # types from the serialized data, so these don't matter.
        super().__init__(CachedSmilesMolHolder(), PatternHolder())
        self.fp_filename = fp_filename
        self.mol_ids = mol_ids

    def GetMatches(  # type: ignore[override]
            self,
            query: str | Chem.Mol,
            recursionPossible: bool = True,
            useChirality: bool = True,
            useQueryQueryMatches: bool = False,
            numThreads: int = -1,
            maxResults: int = -1,
    ) -> pl.DataFrame:
        """Find all molecules matching *query* and return their Papyrus identifiers.

        :param query: a SMILES string or an RDKit :class:`~rdkit.Chem.Mol` query
        :returns: a DataFrame with columns ``idnumber``, ``connectivity``, ``InChIKey``
        """
        if len(self) == 0:
            # GetMatches() raises on an empty library instead of returning no
            # matches - possible when n_shards > molecule count.
            return pl.DataFrame(schema={
                'idnumber': pl.Int64, 'connectivity': pl.Utf8, 'InChIKey': pl.Utf8,
            })
        if isinstance(query, str):
            query = load_molecule(query)
        raw_ids = list(super().GetMatches(
            query=query,
            recursionPossible=recursionPossible,
            useChirality=useChirality,
            useQueryQueryMatches=useQueryQueryMatches,
            numThreads=numThreads,
            maxResults=maxResults,
        ),
        )
        if not raw_ids:
            return pl.DataFrame(schema={
                'idnumber': pl.Int64, 'connectivity': pl.Utf8, 'InChIKey': pl.Utf8,
            })
        ids = [self.mol_ids[i] for i in raw_ids] if self.mol_ids is not None else raw_ids
        return self._get_mapping(ids)

    def substructure(self, query: str | Chem.Mol) -> pl.DataFrame:
        """Alias for :meth:`GetMatches` with default parameters."""
        return self.GetMatches(query)


# ---------------------------------------------------------------------------
# Sharded substructure library
# ---------------------------------------------------------------------------

class PapyrusShardedSubstructureLibrary:
    """Substructure search across a sharded FPSubSim2 database.

    Wraps ``n_shards`` independent :class:`PapyrusSubstructureLibrary`
    instances. Molecule-to-shard partitioning is disjoint by construction
    (:func:`_shard_for_mol_id`), so per-shard results can simply be
    concatenated - no deduplication needed.

    Instances are typically obtained via :meth:`FPSubSim2.get_substructure_lib`
    rather than constructed directly.
    """

    def __init__(
            self,
            fp_filename: str | Path,
            shard_data: list[bytes],
            shard_mol_ids: list[list[int]],
            max_resident_shards: int | None = None,
    ) -> None:
        """Bind to *fp_filename* and prepare per-shard libraries from *shard_data*.

        :param fp_filename: path to the FPSubSim2 HDF5 database
        :param shard_data: one unpadded, serialized ``SubstructLibrary`` byte
            string per shard
        :param shard_mol_ids: one idnumber-mapping list per shard, aligned
            by index with *shard_data* (see
            :class:`PapyrusSubstructureLibrary`'s ``mol_ids``)
        :param max_resident_shards: keep at most this many shard libraries
            deserialized at once; the rest load/query/discard on demand.
            ``None`` (default) keeps every shard resident.
        """
        _check_optional_deps()
        self.fp_filename = fp_filename
        self._shard_data = shard_data
        self._shard_mol_ids = shard_mol_ids
        self.n_shards = len(shard_data)
        self.max_resident_shards = max_resident_shards
        self._resident: list[PapyrusSubstructureLibrary | None]
        if max_resident_shards is None:
            self._resident = [self._load_shard(i) for i in range(self.n_shards)]
        else:
            self._resident = [None] * self.n_shards

    def _load_shard(self, shard_idx: int) -> PapyrusSubstructureLibrary:
        lib = PapyrusSubstructureLibrary(self.fp_filename, mol_ids=self._shard_mol_ids[shard_idx])
        with BytesIO(self._shard_data[shard_idx]) as stream:
            lib.InitFromStream(stream)
        return lib

    def GetMatches(
            self,
            query: str | Chem.Mol,
            recursionPossible: bool = True,
            useChirality: bool = True,
            useQueryQueryMatches: bool = False,
            numThreads: int = -1,
            maxResults: int = -1,
            parallel: bool = True,
    ) -> pl.DataFrame:
        """Find all molecules matching *query* across every shard.

        :param query: a SMILES string or an RDKit :class:`~rdkit.Chem.Mol` query
        :param parallel: fan out across shards via a thread pool; only takes
            effect when every shard is resident (``max_resident_shards`` was
            ``None`` at construction), otherwise shards are queried one at
            a time regardless
        :returns: a DataFrame with columns ``idnumber``, ``connectivity``, ``InChIKey``
        """
        def _query_shard(shard_idx: int) -> pl.DataFrame:
            lib = self._resident[shard_idx]
            if lib is None:
                lib = self._load_shard(shard_idx)
            return lib.GetMatches(
                query,
                recursionPossible=recursionPossible, useChirality=useChirality,
                useQueryQueryMatches=useQueryQueryMatches, numThreads=numThreads,
                maxResults=maxResults,
            )

        if parallel and self.max_resident_shards is None:
            with ThreadPoolExecutor(max_workers=self.n_shards) as pool:
                frames = list(pool.map(_query_shard, range(self.n_shards)))
        else:
            frames = [_query_shard(i) for i in range(self.n_shards)]

        non_empty = [df for df in frames if df.height > 0]
        if not non_empty:
            return pl.DataFrame(schema={
                'idnumber': pl.Int64, 'connectivity': pl.Utf8, 'InChIKey': pl.Utf8,
            })
        return pl.concat(non_empty)

    def substructure(self, query: str | Chem.Mol) -> pl.DataFrame:
        """Alias for :meth:`GetMatches` with default parameters."""
        return self.GetMatches(query)
