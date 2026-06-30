# -*- coding: utf-8 -*-

"""Substructure and similarity search over the Papyrus dataset."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
import warnings
from abc import ABC
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pystow
import rdkit
from rdkit import Chem
from rdkit.Chem.rdSubstructLibrary import CachedMolHolder, PatternHolder, SubstructLibrary
from tqdm.auto import tqdm

try:
    import cupy
except ImportError as e:
    cupy = e

try:
    import tables as tb
except ImportError as e:
    tb = e

try:
    import FPSim2
    from FPSim2.FPSim2 import FPSim2Engine
    from FPSim2.FPSim2Cuda import FPSim2CudaEngine
    from FPSim2.base import BaseEngine
    from FPSim2.io.backends.base import BaseStorageBackend
    from FPSim2.io.backends.pytables import BATCH_WRITE_SIZE, calc_popcnt_bins_pytables, create_schema
    from FPSim2.io.chem import load_molecule
except ImportError as e:
    FPSim2 = e
    # Stub classes so class definitions below do not fail at import time.
    # Using dedicated stubs (not `object`) avoids MRO conflicts when these are
    # mixed with other bases that already implicitly inherit from `object`.
    class BaseStorageBackend: pass
    class BaseEngine: pass
    class FPSim2Engine: pass
    class FPSim2CudaEngine: pass

from .fingerprint import Fingerprint, MorganFingerprint, get_fp_from_name
from .utils.IO import PapyrusVersion, get_num_rows_in_file, locate_file, process_data_version
from .utils.mol_reader import MolSupplier


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_optional_deps() -> None:
    """Raise :exc:`ImportError` when *tables* or *FPSim2* are unavailable."""
    missing = []
    if isinstance(tb, ImportError):
        missing.append('tables')
    if isinstance(FPSim2, ImportError):
        missing.append('FPSim2')
    if missing:
        raise ImportError(
            'Some required dependencies are missing:\n\t' + ', '.join(missing)
        )
    # Guard against partial FPSim2 imports (placeholder strings would remain).
    for name, obj in [
        ('BaseStorageBackend', BaseStorageBackend),
        ('BaseEngine', BaseEngine),
        ('FPSim2Engine', FPSim2Engine),
        ('FPSim2CudaEngine', FPSim2CudaEngine),
    ]:
        if obj is object:
            raise ImportError(f'FPSim2 component {name!r} could not be loaded.')


def _validate_fingerprints(
        fingerprint: Optional[Union[Fingerprint, List[Fingerprint]]],
) -> List[Fingerprint]:
    """Normalise and validate the *fingerprint* argument.

    Returns a non-empty list of :class:`~fingerprint.Fingerprint` instances.
    Falls back to all derived fingerprint types when *fingerprint* is ``None``.

    :param fingerprint: a single :class:`~fingerprint.Fingerprint`, a list
        thereof, or ``None`` to use every available fingerprint type
    :raises ValueError: if any element is not a :class:`~fingerprint.Fingerprint`
    """
    if fingerprint is None:
        return [fp() for fp in Fingerprint.derived()]
    if not isinstance(fingerprint, list):
        fingerprint = [fingerprint]
    for fp in fingerprint:
        if not isinstance(fp, Fingerprint):
            raise ValueError(f'{fp!r} is not a supported Fingerprint instance.')
    return fingerprint


def _fp_table_path(fp: Fingerprint) -> str:
    """Return the HDF5 path for the fps table of *fp*."""
    return f'/similarity_info/{repr(fp)}/fps'


def _derive_connectivity(props: dict, rdmol: Chem.Mol) -> Tuple[str, str]:
    """Extract or derive ``(connectivity, inchikey)`` from SD-file properties.

    When ``connectivity`` is absent it is derived from the first block of the
    InChIKey. When ``InChIKey`` is absent it is computed from the molecule.

    :param props: result of ``rdmol.GetPropsAsDict()``
    :param rdmol: the RDKit molecule (used as a fallback for InChIKey)
    """
    inchikey = props.get('InChIKey', '') or Chem.MolToInchiKey(rdmol) or ''
    connectivity = props.get('connectivity', '') or (inchikey.split('-')[0] if inchikey else '')
    return connectivity, inchikey


def _decode_bytes_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode any ``object``-typed column that contains bytes values to ``str``."""
    for col, dtype in df.dtypes.items():
        if dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
            )
    return df


def _build_result_df(
        raw_results,
        get_mapping,
        score_col: str,
) -> pd.DataFrame:
    """Convert raw FPSim2 ``(id, score)`` pairs into a labelled DataFrame.

    :param raw_results: iterable of ``(mol_id, score)`` tuples returned by
        an FPSim2 search method
    :param get_mapping: callable that maps a list of integer ids to a
        DataFrame with Papyrus identifiers
    :param score_col: column name to assign to the score values
    :returns: a DataFrame with Papyrus identifiers and the score column, or
        an empty DataFrame with those column names when no results are found
    """
    pairs = list(zip(*raw_results)) if raw_results else []
    if not pairs:
        return pd.DataFrame(columns=['idnumber', 'connectivity', 'InChIKey', score_col])
    ids, scores = list(pairs[0]), list(pairs[1])
    df = get_mapping(ids)
    df[score_col] = scores
    return _decode_bytes_columns(df)


def sort_db_file(filename: str, verbose: bool = False) -> None:
    """Sort an FPSubSim2 HDF5 file by fingerprint popcount.

    Sorting enables the efficient popcount-range pruning used by FPSim2 during
    similarity searches. The operation rewrites the file via a temporary copy.

    :param filename: path to the ``.h5`` FPSubSim2 database
    :param verbose: print progress messages and per-table progress bars
    """
    if verbose:
        print('Optimizing FPSubSim2 file.')

    tmp_filename = filename + '_tmp'
    if os.path.isfile(tmp_filename):
        os.remove(tmp_filename)
    os.rename(filename, tmp_filename)

    filters = tb.Filters(complib='blosc', complevel=1, shuffle=True, bitshuffle=True)
    stats: dict = {'groups': 0, 'leaves': 0, 'links': 0, 'bytes': 0, 'hardlinks': 0}

    with tb.open_file(tmp_filename, mode='r') as src:
        with tb.open_file(filename, mode='w') as dst:
            siminfo_group = dst.create_group(
                dst.root, 'similarity_info', 'Infos for similarity search'
            )
            simfp_groups = list(src.walk_groups('/similarity_info/'))

            for i, simfp_group in enumerate(simfp_groups):
                if not simfp_group._v_name:
                    continue
                dst_group = simfp_group._f_copy(
                    siminfo_group, recursive=False, filters=filters, stats=stats
                )
                fp_tables = list(src.iter_nodes(simfp_group, classname='Table'))
                table_iter = (
                    tqdm(fp_tables,
                         desc=f'Optimizing tables of group ({i}/{len(simfp_groups)})',
                         leave=False
                         )
                    if verbose else fp_tables
                )
                for fp_table in table_iter:
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
                    popcounts = dst.create_vlarray(
                        dst_group, 'popcounts', tb.ObjectAtom(),
                        f'Popcounts of {dst_group._v_name}'
                    )
                    for x in popcnt_bins:
                        popcounts.append(x)

            if verbose:
                print('Optimizing remaining groups and arrays.')
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

    if verbose:
        print('Cleaning up temporary files.')
    os.remove(tmp_filename)


# ---------------------------------------------------------------------------
# FPSubSim2 — database creation and loading
# ---------------------------------------------------------------------------

class FPSubSim2:
    """Create, load, and extend a multi-fingerprint similarity + substructure
    search database backed by HDF5 (via PyTables).

    The database stores:

    * one or more fingerprint tables (for Tanimoto / Tversky similarity)
    * a serialised RDKit ``SubstructLibrary`` (for exact subgraph isomorphism)
    * a molecule-ID → ``(connectivity, InChIKey)`` mapping table
    """

    def __init__(self) -> None:
        _check_optional_deps()
        self.version: Optional[PapyrusVersion] = None
        self.is3d: Optional[bool] = None
        self.sd_file: Optional[str] = None
        self.h5_filename: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def create_from_papyrus(
            self,
            is3d: bool = False,
            version: Union[str, PapyrusVersion] = 'latest',
            outfile: Optional[str] = None,
            fingerprint: Optional[Union[Fingerprint, List[Fingerprint]]] = None,
            root_folder: Optional[str] = None,
            progress: bool = True,
            njobs: int = 1,
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
        """
        if fingerprint is None:
            fingerprint = MorganFingerprint()

        self.version = PapyrusVersion(version=version)
        if not self.version.is_downloaded(root_folder=root_folder):
            raise ValueError(f'Version {self.version.version} not found. Did you download it first?')
        self.is3d = is3d

        if root_folder is not None:
            os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)

        structure_dir = pystow.join(
            'papyrus', self.version.version_old_fmt, 'structures'
        )
        filenames = locate_file(
            structure_dir.as_posix(),
            rf'\d+\.\d+_combined_{3 if is3d else 2}D_set_'
            rf'with{"out" if not is3d else ""}_stereochemistry\.sd.*',
        )
        self.sd_file = filenames[0]
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
        )

    def create(
            self,
            sd_file: str,
            outfile: Optional[str] = None,
            fingerprint: Union[Fingerprint, List[Fingerprint]] = None,
            progress: bool = True,
            total: Optional[int] = None,
            njobs: int = 1,
    ) -> None:
        """Create an FPSubSim2 database from any SD file.

        :param sd_file: path to the SD file containing structures
        :param outfile: output ``.h5`` path; auto-generated when ``None``
        :param fingerprint: fingerprint(s) to store; defaults to all available
        :param progress: display progress bars
        :param total: molecule count for progress bar display
        :param njobs: worker processes (``-1`` = all, ``1`` = single-process)
        :raises ValueError: if *njobs* is invalid
        """
        self.sd_file = sd_file
        fingerprint = _validate_fingerprints(fingerprint)

        dim_tag = '3D' if self.is3d else '2D'
        version_str = str(self.version) if self.version is not None else 'custom'
        self.h5_filename = outfile or f'Papyrus_{version_str}_FPSubSim2_{dim_tag}.h5'

        if not isinstance(njobs, int) or njobs < -1:
            raise ValueError('njobs must be -1 or a positive integer.')

        filters = tb.Filters()
        with tb.open_file(self.h5_filename, mode='w') as h5file:
            simil_group = h5file.create_group(
                h5file.root, 'similarity_info', 'Infos for similarity search'
            )
            subst_group = h5file.create_group(
                h5file.root, 'substructure_info', 'Infos for substructure search'
            )
            h5file.create_earray(
                subst_group, 'substruct_lib', tb.UInt64Atom(), (0,),
                'Substructure search library',
            )
            h5file.create_table(
                h5file.root, 'mol_mappings',
                np.dtype([
                    ('idnumber', '<i8'),
                    ('connectivity', 'S14'),
                    ('InChIKey', 'S27'),
                ]
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
                self.version.version_old_fmt if self.version is not None else '',
                dim_tag,
            ]
            )
            for fp_type in fingerprint:
                fp_group = h5file.create_group(
                    simil_group, repr(fp_type), f'Similarity {repr(fp_type)}'
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

        if njobs in (0, 1):
            self._single_process_create(fingerprint, progress, total)
        else:
            self._parallel_create(njobs, fingerprint, progress, total)

        self.load(self.h5_filename)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, fpsubsim_path: str) -> None:
        """Load an existing FPSubSim2 database file.

        :param fpsubsim_path: path to the ``.h5`` database
        :raises ValueError: if *fpsubsim_path* does not exist
        """
        if not os.path.isfile(fpsubsim_path):
            raise ValueError(f'File does not exist: {fpsubsim_path!r}')
        self.h5_filename = fpsubsim_path

        with tb.open_file(self.h5_filename) as h5file:
            rdkit_version, version_str, dim_tag = h5file.root.config.read()[0]

        if rdkit.__version__ != rdkit_version:
            warnings.warn(
                f'RDKit version mismatch: library was built with {rdkit_version}, '
                f'current version is {rdkit.__version__}. '
                'Consider regenerating the FPSubSim2 library to avoid unexpected behaviour.'
            )

        self.is3d = (dim_tag == '3D')
        # Reconstruct the PapyrusVersion from the stored old-format string.
        # If the string is empty (database built from a custom SD file), leave as None.
        self.version = PapyrusVersion(version=version_str) if version_str else None
        # Invalidate the fingerprint cache so the next access re-reads the file.
        self._avail_fp: Optional[dict] = None

    # ------------------------------------------------------------------
    # Internal: single-process creation
    # ------------------------------------------------------------------

    def _single_process_create(
            self,
            fingerprint: List[Fingerprint],
            progress: bool,
            total: Optional[int],
    ) -> None:
        """Populate similarity and substructure tables from a single process."""
        with tb.open_file(self.h5_filename, mode='r+') as h5file:
            lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
            subst_table = h5file.root.substructure_info.substruct_lib
            mappings_table = h5file.root.mol_mappings
            # Map each fingerprint repr → HDF5 table path (consistent key everywhere).
            table_paths = {repr(fp): _fp_table_path(fp) for fp in fingerprint}
            fps = defaultdict(list)
            mappings = []

            with MolSupplier(source=self.sd_file, total=total,
                             show_progress=progress, start_id=1
                             ) as supplier:
                for mol_id, rdmol in supplier:
                    lib.AddMol(rdmol)
                    connectivity, inchikey = _derive_connectivity(rdmol.GetPropsAsDict(), rdmol)
                    mappings.append((mol_id, connectivity, inchikey))

                    for fp_type in fingerprint:
                        fps[repr(fp_type)].append((mol_id, *fp_type.get(rdmol)))

                    if len(fps[repr(fingerprint[0])]) == BATCH_WRITE_SIZE:
                        for fp_type in fingerprint:
                            h5file.get_node(table_paths[repr(fp_type)]).append(
                                fps[repr(fp_type)]
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

            lib_bytes = _serialize_substruct_lib(lib)
            subst_table.attrs.padding = _padding_for(lib_bytes)
            lib_bytes = _pad_to_int64(lib_bytes)
            subst_table.append(np.frombuffer(lib_bytes, dtype=np.int64))

        sort_db_file(self.h5_filename, verbose=progress)

    # ------------------------------------------------------------------
    # Internal: parallel creation
    # ------------------------------------------------------------------

    def _parallel_create(
            self,
            njobs: int,
            fingerprint: List[Fingerprint],
            progress: bool,
            total: Optional[int],
    ) -> None:
        """Populate similarity and substructure tables using multiple processes."""
        # Pass only (type, params) pairs to worker processes — instances are
        # not safe to share across process boundaries.
        fp_specs = [(type(fp), fp.params) for fp in fingerprint]
        table_paths = {repr(fp): _fp_table_path(fp) for fp in fingerprint}

        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()

        n_workers = (multiprocessing.cpu_count() - 2) if njobs == -1 else max(njobs - 1, 1)

        reader = multiprocessing.Process(
            target=_reader_process,
            args=(self.sd_file, n_workers, total, False, input_queue),
        )
        writer = multiprocessing.Process(
            target=_writer_process,
            args=(self.h5_filename, output_queue, table_paths, total, progress),
        )
        workers = [
            multiprocessing.Process(
                target=_worker_process,
                args=(fp_specs, input_queue, output_queue),
            )
            for _ in range(n_workers)
        ]

        reader.start()
        writer.start()
        for w in workers:
            w.start()

        active = [reader] + workers
        while active:
            active[0].join(10)
            if not active[0].is_alive():
                del active[0]

        output_queue.put('STOP')
        writer.join()

        input_queue.close();
        input_queue.join_thread()
        output_queue.close();
        output_queue.join_thread()

        sort_db_file(self.h5_filename, verbose=progress)

    # ------------------------------------------------------------------
    # Fingerprint and substructure library access
    # ------------------------------------------------------------------

    @property
    def available_fingerprints(self) -> dict:
        """Dict mapping fingerprint signature → :class:`~fingerprint.Fingerprint`
        instance for every fingerprint stored in the database.

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

    def get_substructure_lib(self) -> 'PapyrusSubstructureLibrary':
        """Return the deserialized RDKit substructure library.

        :raises ValueError: if the database file does not exist yet
        """
        if not os.path.isfile(self.h5_filename):
            raise ValueError('Database file must be created first.')
        with tb.open_file(self.h5_filename, mode='r') as h5file:
            padding = h5file.root.substructure_info.substruct_lib.attrs.padding
            data = h5file.root.substructure_info.substruct_lib.read()
        raw = data.tobytes('C')
        raw = raw[:-padding] if padding else raw
        with BytesIO(raw) as stream:
            lib = PapyrusSubstructureLibrary(self.h5_filename)
            lib.InitFromStream(stream)
        return lib

    def get_similarity_lib(
            self,
            fp_signature: Optional[str] = None,
            cuda: bool = False,
    ) -> Union['FPSubSim2Engine', 'FPSubSim2CudaEngine']:
        """Return a similarity search engine for the requested fingerprint.

        :param fp_signature: signature of the fingerprint to use; defaults to
            the first available fingerprint when ``None``
        :param cuda: use the GPU-accelerated engine
        :raises ValueError: if the database does not exist or the signature is
            not found in the database
        """
        if not os.path.isfile(self.h5_filename):
            raise ValueError('Database file must be created first.')
        available = self.available_fingerprints
        if fp_signature is None:
            fp_signature = next(iter(available))
        if fp_signature not in available:
            raise ValueError(
                f'Fingerprint {fp_signature!r} not available. '
                f'Choose one of: {list(available)}'
            )
        engine_cls = FPSubSim2CudaEngine if cuda else FPSubSim2Engine
        return engine_cls(self.h5_filename, fp_signature)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_fingerprint(
            self,
            fingerprint: Fingerprint,
            papyrus_sd_file: str,
            progress: bool = True,
            total: Optional[int] = None,
    ) -> None:
        """Add a new fingerprint type to the database.

        :param fingerprint: the :class:`~fingerprint.Fingerprint` to add
        :param papyrus_sd_file: SD file containing the structures
        :param progress: display a progress bar
        :param total: molecule count for the progress bar
        """
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
            total: Optional[int] = None,
    ) -> None:
        """Append new molecules to all fingerprint tables and the substructure library.

        :param papyrus_sd_file: SD file containing new structures to add
        :param progress: display a progress bar
        :param total: molecule count for the progress bar
        """
        for signature, fingerprint in self.available_fingerprints.items():
            backend = PyTablesMultiFpStorageBackend(self.h5_filename, signature)
            backend.append_fps(
                MolSupplier(source=papyrus_sd_file),
                total=total,
                progress=progress,
                sort=False,
            )

        substruct_lib = self.get_substructure_lib()
        with MolSupplier(source=papyrus_sd_file, total=total, show_progress=progress) as supplier:
            for _, rdmol in supplier:
                substruct_lib.AddMol(rdmol)

        lib_bytes = _pad_to_int64(_serialize_substruct_lib(substruct_lib))
        padding = _padding_for(lib_bytes)
        lib_ints = np.frombuffer(lib_bytes, dtype=np.int64)

        with tb.open_file(self.h5_filename, mode='a') as h5file:
            h5file.remove_node(h5file.root.substructure_info.substruct_lib)
            arr = h5file.create_earray(
                h5file.root.substructure_info, 'substruct_lib',
                tb.UInt64Atom(), (0,), 'Substructure search library',
            )
            arr.attrs.padding = padding
            arr.append(lib_ints)

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
        n_workers: int,
        total: Optional[int],
        progress: bool,
        input_queue: multiprocessing.Queue,
) -> None:
    """Read molecules from *sd_file* and feed them to the worker queue."""
    with MolSupplier(source=sd_file, total=total, show_progress=progress, start_id=1) as supplier:
        count = 0
        for mol_id, rdmol in supplier:
            input_queue.put((mol_id, rdmol, rdmol.GetPropsAsDict()))
            count += 1
            # Back-pressure: prevent the queue from growing without bound.
            if count > BATCH_WRITE_SIZE * n_workers * 1.5:
                while input_queue.qsize() > BATCH_WRITE_SIZE:
                    time.sleep(10)
                count = 0
    # Send one termination sentinel per worker.
    for _ in range(n_workers):
        input_queue.put('END')


def _worker_process(
        fp_specs: List[Tuple[type, dict]],
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
) -> None:
    """Compute fingerprints for each molecule and forward results to the writer."""
    while True:
        data = input_queue.get()
        if data == 'END':
            break
        mol_id, rdmol, props = data
        output_queue.put(('substructure', rdmol))

        connectivity, inchikey = _derive_connectivity(props, rdmol)
        output_queue.put(('mappings', (mol_id, connectivity, inchikey)))

        for fp_cls, fp_params in fp_specs:
            fper = fp_cls(**fp_params)
            output_queue.put(('similarity', repr(fper), (mol_id, *fper.get(rdmol))))


def _writer_process(
        h5_filename: str,
        output_queue: multiprocessing.Queue,
        table_paths: dict,
        total: Optional[int],
        progress: bool,
) -> None:
    """Consume results from worker processes and write them to the HDF5 file."""
    lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
    pbar = tqdm(total=total, smoothing=0.0) if progress else None

    mappings_insert = []
    similarity_insert = defaultdict(list)

    with tb.open_file(h5_filename, mode='r+') as h5file:
        while True:
            data = output_queue.get()
            if data == 'STOP':
                # Flush remaining data.
                h5file.root.mol_mappings.append(mappings_insert)
                for fp_id, fp_rows in similarity_insert.items():
                    h5file.get_node(table_paths[fp_id]).append(fp_rows)
                # Serialise and store the substructure library.
                lib_bytes = _pad_to_int64(_serialize_substruct_lib(lib))
                padding = _padding_for(lib_bytes)
                lib_ints = np.frombuffer(lib_bytes, dtype=np.int64)
                h5file.root.substructure_info.substruct_lib.attrs.padding = padding
                h5file.root.substructure_info.substruct_lib.append(lib_ints)
                # Create indices.
                for path in table_paths.values():
                    h5file.get_node(path).cols.popcnt.create_index(kind='full')
                h5file.root.mol_mappings.cols.idnumber.create_index(kind='full')
                break

            if data[0] == 'mappings':
                mappings_insert.append(data[1])
                if pbar is not None:
                    pbar.update()
            elif data[0] == 'substructure':
                lib.AddMol(data[1])
            elif data[0] == 'similarity':
                _, fp_id, fp_row = data
                similarity_insert[fp_id].append(fp_row)

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

    if pbar is not None:
        pbar.close()

    with tb.open_file(h5_filename, mode='r+') as h5file:
        h5file.root.mol_mappings.cols.idnumber.reindex()


# ---------------------------------------------------------------------------
# PyTables storage backend
# ---------------------------------------------------------------------------

class PyTablesMultiFpStorageBackend(BaseStorageBackend):
    """PyTables-backed storage backend for multi-fingerprint FPSubSim2 databases."""

    def __init__(
            self,
            fp_filename: str,
            fp_signature: str,
            in_memory_fps: bool = True,
            fps_sort: bool = False,
    ) -> None:
        super().__init__(fp_filename)
        self.name = 'pytables'

        with tb.open_file(self.fp_filename, mode='r') as fp_file:
            self._fp_table_mappings: Dict[str, List[str]] = {}
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
                f'Choose one of: {", ".join(self._fp_table_mappings)}'
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

    def read_parameters(self) -> Tuple[str, dict, str]:
        """Read fingerprint metadata for the currently selected fingerprint."""
        with tb.open_file(self.fp_filename, mode='r') as fp_file:
            rdkit_ver = fp_file.root.config[0]
            fp_table = fp_file.get_node(self._current_fp_path)
            fp_type = fp_table.attrs.fp_type
            fp_params = json.loads(fp_table.attrs.fp_params)
        return fp_type, fp_params, rdkit_ver

    def get_fps_chunk(self, chunk_range: Tuple[int, int]) -> np.ndarray:
        with tb.open_file(self.fp_filename, mode='r') as fp_file:
            return fp_file.get_node(self._current_fp_path)[slice(*chunk_range)]

    def load_popcnt_bins(self, fps_sort: bool) -> None:
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

    def delete_fps(self, ids_list: List[int]) -> None:
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
            total: Optional[int] = None,
            sort: bool = True,
    ) -> None:
        """Append fingerprints for molecules from *supplier* to the database."""
        with tb.open_file(self.fp_filename, mode='a') as fp_file:
            fps_table = fp_file.get_node(self._current_fp_path)
            start_id = max(
                (row['fp_id'] for row in fps_table.iterrows()), default=1
            )
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
              'Call `append_fps` to populate it.'
              )


# ---------------------------------------------------------------------------
# Mapping helper mixin
# ---------------------------------------------------------------------------

class _MappingMixin:
    """Mixin that provides molecule-ID → Papyrus-identifier lookup."""

    fp_filename: str  # supplied by concrete subclass

    def _get_mapping(self, ids: Union[List[int], int]) -> pd.DataFrame:
        """Return a DataFrame with Papyrus identifiers for the given integer *ids*.

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
            rows = []
            for i in ids:
                ptr = mappings_table.where(f'idnumber == {i}')
                try:
                    rows.append(next(ptr).fetch_all_fields())
                except StopIteration:
                    raise ValueError(f'Index {i} not found in the database.')

        df = pd.DataFrame.from_records(rows, columns=colnames)
        return _decode_bytes_columns(df)


# ---------------------------------------------------------------------------
# Similarity search engines
# ---------------------------------------------------------------------------

class BaseMultiFpEngine(_MappingMixin, BaseEngine, ABC):
    """Abstract base for CPU and GPU similarity search engines."""

    def __init__(
            self,
            fp_filename: str,
            fp_signature: str,
            storage_backend: str,
            in_memory_fps: bool,
            fps_sort: bool,
    ) -> None:
        self.fp_filename = fp_filename
        self.in_memory_fps = in_memory_fps
        if storage_backend == 'pytables':
            self.storage = PyTablesMultiFpStorageBackend(
                fp_filename, fp_signature,
                in_memory_fps=in_memory_fps,
                fps_sort=fps_sort,
            )

    def load_query(self, query_string: str) -> np.ndarray:
        """Parse a SMILES, InChI, or molblock query string into a fingerprint array.

        :param query_string: SMILES, InChI, or molblock
        :raises ValueError: if the molecule cannot be parsed
        """
        rdmol = load_molecule(query_string)
        if rdmol is None:
            raise ValueError(f'Could not parse query molecule: {query_string!r}')
        fp = get_fp_from_name(self.storage.fp_type, **self.storage.fp_params).get(rdmol)
        return np.array((0, *fp), dtype=np.uint64)


class FPSubSim2Engine(BaseMultiFpEngine, FPSim2Engine):
    """CPU-based similarity search engine for FPSubSim2 databases."""

    def __init__(
            self,
            fp_filename: str,
            fp_signature: str,
            in_memory_fps: bool = True,
            fps_sort: bool = False,
            storage_backend: str = 'pytables',
    ) -> None:
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

    def similarity(self, query_string: str, threshold: float, n_workers: int = 1) -> pd.DataFrame:
        """In-memory Tanimoto similarity search."""
        raw = FPSim2Engine.similarity(self, query_string, threshold, n_workers)
        return _build_result_df(raw, self._get_mapping, self._score_col('Tanimoto', threshold))

    def on_disk_similarity(
            self, query_string: str, threshold: float,
            n_workers: int = 1, chunk_size: int = 0,
    ) -> pd.DataFrame:
        """On-disk Tanimoto similarity search."""
        raw = FPSim2Engine.on_disk_similarity(self, query_string, threshold, n_workers, chunk_size)
        return _build_result_df(raw, self._get_mapping, self._score_col('Tanimoto', threshold))

    def tversky(
            self, query_string: str, threshold: float,
            a: float, b: float, n_workers: int = 1,
    ) -> pd.DataFrame:
        """In-memory Tversky similarity search."""
        raw = FPSim2Engine.tversky(self, query_string, threshold, a, b, n_workers)
        return _build_result_df(raw, self._get_mapping, self._score_col('Tversky', threshold))

    def on_disk_tversky(
            self, query_string: str, threshold: float,
            a: float, b: float,
            n_workers: int = 1, chunk_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """On-disk Tversky similarity search."""
        raw = FPSim2Engine.on_disk_tversky(self, query_string, threshold, a, b, n_workers, chunk_size)
        return _build_result_df(raw, self._get_mapping, self._score_col('Tversky', threshold))

    def substructure(self, query_string: str, n_workers: int = 1):
        raise NotImplementedError(
            'Use the FPSubSim2 substructure library (get_substructure_lib) '
            'for exact subgraph isomorphism.'
        )

    def on_disk_substructure(self, query_string: str, n_workers: int = 1, chunk_size: int = None):
        raise NotImplementedError(
            'Use the FPSubSim2 substructure library (get_substructure_lib) '
            'for exact subgraph isomorphism.'
        )


class FPSubSim2CudaEngine(BaseMultiFpEngine, FPSim2CudaEngine):
    """GPU-accelerated similarity search engine for FPSubSim2 databases."""

    def __init__(
            self,
            fp_filename: str,
            fp_signature: str,
            storage_backend: str = 'pytables',
            kernel: str = 'raw',
    ) -> None:
        if isinstance(cupy, ImportError):
            raise ImportError('cupy is required for GPU-accelerated search.')
        if kernel not in ('raw', 'element_wise'):
            raise ValueError("kernel must be 'raw' or 'element_wise'.")
        super().__init__(
            fp_filename=fp_filename,
            fp_signature=fp_signature,
            storage_backend=storage_backend,
            in_memory_fps=True,
            fps_sort=False,
        )
        self.kernel = kernel
        if kernel == 'raw':
            self.cuda_db = cupy.asarray(self.fps[:, 1:-1])
            self.cuda_ids = cupy.asarray(self.fps[:, 0])
            self.cuda_popcnts = cupy.asarray(self.fps[:, -1])
            self.cupy_kernel = cupy.RawKernel(
                self.raw_kernel.format(block=self.cuda_db.shape[1]),
                name='taniRAW',
                options=('-std=c++14',),
            )
        else:
            self.cuda_db = cupy.asarray(self.fps)
            self.cupy_kernel = cupy.ElementwiseKernel(
                in_params='raw T db, raw U query, uint64 in_width, float32 threshold',
                out_params='raw V out',
                operation=self.ew_kernel,
                name='taniEW',
                options=('-std=c++14',),
                reduce_dims=False,
            )

    def similarity(self, query_string: str, threshold: float) -> pd.DataFrame:
        """GPU Tanimoto similarity search."""
        raw = FPSim2CudaEngine.similarity(self, query_string, threshold)
        return _build_result_df(
            raw, self._get_mapping,
            f'Tanimoto > {threshold} ({self.storage._current_fp})',
        )


# ---------------------------------------------------------------------------
# Substructure library with Papyrus identifier mapping
# ---------------------------------------------------------------------------

class PapyrusSubstructureLibrary(_MappingMixin, SubstructLibrary):
    """RDKit ``SubstructLibrary`` extended with Papyrus identifier lookup.

    Instances are typically obtained via :meth:`FPSubSim2.get_substructure_lib`
    rather than constructed directly.

    .. note::
        The ``fp_filename`` parameter points to the FPSubSim2 HDF5 database,
        which stores the ``mol_mappings`` table used by :meth:`GetMatches`.
    """

    def __init__(self, fp_filename: str) -> None:
        # Initialise SubstructLibrary with the standard molecule holders.
        super().__init__(CachedMolHolder(), PatternHolder())
        self.fp_filename = fp_filename

    def GetMatches(
            self,
            query: Union[str, Chem.Mol],
            recursionPossible: bool = True,
            useChirality: bool = True,
            useQueryQueryMatches: bool = False,
            numThreads: int = -1,
            maxResults: int = -1,
    ) -> pd.DataFrame:
        """Find all molecules matching *query* and return their Papyrus identifiers.

        :param query: a SMILES string or an RDKit :class:`~rdkit.Chem.Mol` query
        :returns: a DataFrame with columns ``idnumber``, ``connectivity``, ``InChIKey``
        """
        if isinstance(query, str):
            query = load_molecule(query)
        ids = list(super().GetMatches(
            query=query,
            recursionPossible=recursionPossible,
            useChirality=useChirality,
            useQueryQueryMatches=useQueryQueryMatches,
            numThreads=numThreads,
            maxResults=maxResults,
        )
        )
        if not ids:
            return pd.DataFrame(columns=['idnumber', 'connectivity', 'InChIKey'])
        return self._get_mapping(ids)

    def substructure(self, query: Union[str, Chem.Mol]) -> pd.DataFrame:
        """Alias for :meth:`GetMatches` with default parameters."""
        return self.GetMatches(query)
