# -*- coding: utf-8 -*-

from __future__ import annotations

import multiprocessing
import os
import time
import warnings
from collections import defaultdict
from io import BytesIO
from typing import Optional, Tuple, Union

import pandas as pd
import pystow
import rdkit
from rdkit.Chem.rdSubstructLibrary import SubstructLibrary, PatternHolder, CachedMolHolder
from tqdm import tqdm

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
    from FPSim2.io.backends.pytables import create_schema, BATCH_WRITE_SIZE, calc_popcnt_bins_pytables
    from FPSim2.io.backends.base import BaseStorageBackend
    from FPSim2.base import BaseEngine
    from FPSim2.FPSim2 import FPSim2Engine
    from FPSim2.FPSim2Cuda import FPSim2CudaEngine
    from FPSim2.io.chem import load_molecule
except ImportError as e:
    FPSim2 = e
    # Placeholders
    BaseStorageBackend = str
    BaseEngine = str
    FPSim2Engine = str
    FPSim2CudaEngine = str


from .fingerprint import *
from .utils.mol_reader import MolSupplier
from .utils.IO import locate_file, get_num_rows_in_file, process_data_version


class FPSubSim2:
    def __init__(self):
        if isinstance(tb, ImportError) and isinstance(FPSim2, ImportError):
            raise ImportError('Some required dependencies are missing:\n\ttables, FPSim2')
        elif isinstance(tb, ImportError):
            raise ImportError('Some required dependencies are missing:\n\ttables')
        elif isinstance(FPSim2, ImportError):
            raise ImportError('Some required dependencies are missing:\n\tFPSim2')
        elif isinstance(BaseStorageBackend, str) or \
                isinstance(BaseEngine, str) or \
                isinstance(FPSim2Engine, str) or \
                isinstance(FPSim2CudaEngine, str):
            raise ImportError('Some FPSim2 components could not be loaded')
        self.version = None
        self.is3d = None
        self.sd_file = None
        self.h5_filename = None

    def create_from_papyrus(self,
                            is3d: bool = False,
                            version: str = 'latest',
                            outfile: Optional[str] = None,
                            fingerprint: Optional[Union[Fingerprint, List[Fingerprint]]] = MorganFingerprint(),
                            root_folder: Optional[str] = None,
                            progress: bool = True,
                            njobs: int = 1):
        """Create an extended FPSim2 database from Papyrus data.

        :param is3d: Toggle the use of non-standardised (3D) data (default: False)
        :param version: version of the Papyrus dataset to be used
        :param outfile: filename or filepath of output database
        :param fingerprint: fingerprints to be calculated, if None uses all available
        :param root_folder: folder containing the bioactivity dataset (default: pystow's home folder)
        :param progress: whether progress should be shown
        :param njobs: number of concurrent processes (-1 for all available logical cores)
        :return:
        """
        # Set version
        self.version = process_data_version(version=version, root_folder=root_folder)
        # Set 3D
        self.is3d = is3d
        # Determine default paths
        if root_folder is not None:
            os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
        source_path = pystow.join('papyrus', self.version, 'structures')
        # Find the file
        filenames = locate_file(source_path.as_posix(),
                                rf'\d+\.\d+_combined_{3 if is3d else 2}D_set_'
                                rf'with{"out" if not is3d else ""}_stereochemistry\.sd.*')
        self.sd_file = filenames[0]
        total = get_num_rows_in_file(filetype='structures', is3D=is3d, version=self.version, root_folder=root_folder)
        self.h5_filename = outfile
        self.create(sd_file=self.sd_file, outfile=outfile, fingerprint=fingerprint,
                    total=total, progress=progress, njobs=njobs)

    def create(self,
               sd_file: str,
               outfile: Optional[str] = None,
               fingerprint: Union[Fingerprint, List[Fingerprint]] = MorganFingerprint(),
               progress: bool = True,
               total: Optional[int] = None,
               njobs: int = 1):
        """Create an extended FPSim2 database to deal with multiple similarity
        fingerprints and handle full substructure search (subgraph isomorphism)
        and load it when finished.

        :param sd_file: sd file containing chemical structures
        :param outfile: filename or filepath of output database
        :param fingerprint: fingerprints to be calculated; if None, uses all available
        :param progress: whether progress should be shown
        :param total: number of molecules for progress display
        :param njobs: number of concurrent processes (-1 for all available logical cores)
        """
        self.sd_file = sd_file
        # Set outfile name if not supplied
        if outfile is None:
            self.h5_filename = f'Papyrus_{self.version}_FPSubSim2_{3 if self.is3d else 2}D.h5'
        else:
            self.h5_filename = outfile
        # Set fingerprints if not supplied
        if fingerprint is not None:
            if isinstance(fingerprint, list):
                for x in fingerprint:
                    if not isinstance(x, Fingerprint):
                        raise ValueError(f'{x} is not a supported fingerprint')
            elif not isinstance(fingerprint, Fingerprint):
                raise ValueError(f'{fingerprint} is not a supported fingerprint')
        else:
            fingerprint = [fp() for fp in Fingerprint.derived()]
        if not isinstance(fingerprint, list):
            fingerprint = [fingerprint]
        # Ensure njobs is a correct value
        if not isinstance(njobs, int) or njobs < -1:
            raise ValueError('number of jobs must be -1 or above')
        # set compression
        filters = tb.Filters()
        # set the output file and fps table
        with tb.open_file(self.h5_filename, mode="w") as h5file:
            # group to hold similarity tables
            simil_group = h5file.create_group(
                h5file.root, "similarity_info", "Infos for similarity search")
            # group to hold substructure library
            subst_group = h5file.create_group(
                h5file.root, "substructure_info", "Infos for substructure search")
            # Array containing processed binary of the substructure library
            _ = h5file.create_earray(
                subst_group, 'substruct_lib', tb.UInt64Atom(), (0,), 'Substructure search library')
            # Table for mapping indices to identifiers
            h5file.create_table(
                h5file.root, 'mol_mappings',
                np.dtype([("idnumber", "<i8"), ("connectivity", "S14"), ("InChIKey", "S27")]),
                'Molecular mappings', expectedrows=1300000, filters=filters)
            # Set config table containing  rdkit version
            param_table = h5file.create_vlarray(
                h5file.root, "config", atom=tb.ObjectAtom())
            param_table.append([rdkit.__version__, self.version, '3D' if self.is3d else '2D'])
            # Create fingerprint tables
            for fp_type in fingerprint:
                fp_group = h5file.create_group(simil_group, repr(fp_type), f'Similarity {repr(fp_type)}')
                particle = create_schema(fp_type.length)
                fp_table = h5file.create_table(fp_group, 'fps', particle, 'Similarity FPs', expectedrows=1300000,
                                               filters=filters)
                fp_table.attrs.fp_type = fp_type.name
                fp_table.attrs.fp_id = repr(fp_type)
                fp_table.attrs.length = fp_type.length
                fp_table.attrs.fp_params = json.dumps(fp_type.params)
        # Get the number of molecules to process
        if njobs in [0, 1]:
            self._single_process_create(fingerprint, progress, total)
        else:
            self._parallel_create(njobs, fingerprint, progress, total)
        self.load(self.h5_filename)

    def load(self, fpsubsim_path: str):
        """Load an extended FPSim2 database to deal with multiple similarity
        fingerprints and handle full substructure search (subgraph isomorphism).

        :param fpsubsim_path: path to the FPSubSim2 library
        """
        if not os.path.isfile(fpsubsim_path):
            raise ValueError(f'File {fpsubsim_path} does not exist')
        self.h5_filename = fpsubsim_path
        with tb.open_file(self.h5_filename) as h5file:
            rdkit_version, self.version, is3D = h5file.root.config.read()[0]
        if rdkit.__version__ != rdkit_version:
            warnings.warn(f'RDKit version {rdkit.__version__} differs: library was generated with {rdkit_version}. '
                          'Consider regenerating the FPSubSim2 library to avoid unexpected behaviour.')
        self.is3d = is3D == "3D"

    def _single_process_create(self, fingerprint: Union[Fingerprint, List[Fingerprint]], progress: bool = True,
                               total: Optional[int] = None):
        """Fill in the similarity tables from a unique process."""
        with tb.open_file(self.h5_filename, mode="r+") as h5file:
            # substructure rdkit library
            lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
            # Links to goups and tables
            subst_table = h5file.root.substructure_info.substruct_lib
            mappings_table = h5file.root.mol_mappings
            # Create fingerprints, mappings and substructure library
            table_paths = {}  # path to fp tables
            fps = defaultdict(list)  # data to be written into fp tables
            mappings = []
            for fp_type in fingerprint:
                table_paths[repr(fp_type)] = f"/similarity_info/{repr(fp_type)}/fps"
            with MolSupplier(source=self.sd_file, total=total, show_progress=progress,
                             start_id=1) as supplier:
                for mol_id, rdmol in supplier:
                    # Add molecule to substructure search lib
                    lib.AddMol(rdmol)
                    # Get mapping information
                    props = rdmol.GetPropsAsDict()
                    connectivity = props.get('connectivity', '')
                    inchikey = props.get('InChIKey', Chem.MolToInchiKey(rdmol))
                    if not connectivity:
                        connectivity = inchikey.split('-')[0]
                    mappings.append((mol_id, connectivity, inchikey))
                    for fp_type in fingerprint:
                        # generate fingerprint
                        fp = fp_type.get(rdmol)
                        fps[fp_type].append((mol_id, *fp))
                        # flush buffer
                    if len(fps[fingerprint[0]]) == BATCH_WRITE_SIZE:
                        for fp_type in fingerprint:
                            h5file.get_node(table_paths[str(fp_type)]).append(fps[fp_type])
                            mappings_table.append(mappings)
                        fps, mappings = defaultdict(list), []
                # append last batch < 32k
                if len(fps[fingerprint[0]]):
                    for fp_type in fingerprint:
                        h5file.get_node(table_paths[str(fp_type)]).append(fps[fp_type])
                        h5file.get_node(table_paths[str(fp_type)]).flush()
                    mappings_table.append(mappings)
                    mappings_table.flush()

            # create index so table can be sorted
            for fp_type in fingerprint:
                h5file.get_node(table_paths[str(fp_type)]).cols.popcnt.create_index(kind="full")
            h5file.root.mol_mappings.cols.idnumber.create_index(kind="full")

            # serialize substruct lib and pad
            lib_bytes = lib.Serialize()
            remainder = len(lib_bytes) % 8
            padding = 8 - remainder if remainder else 0  # int64 are 8 bytes
            if padding:
                lib_bytes += b'\x00' * padding
            lib_ints = np.frombuffer(lib_bytes, dtype=np.int64)
            # save into h5
            subst_table.attrs.padding = padding
            subst_table.append(lib_ints)
        # sort by popcounts
        sort_db_file(self.h5_filename, verbose=progress)

    def _parallel_create(self, njobs=-1, fingerprint: Union[Fingerprint, List[Fingerprint]] = None,
                         progress: bool = True, total: Optional[int] = None):
        """Fill in the similarity tables with multiple processes."""
        # Fingerprint types and params to be passed to workers (instances are not thread safe)
        fp_types = [(type(fp_type), fp_type.params) for fp_type in fingerprint]
        # Mappings from fingerprint id to table path
        table_paths = {repr(fp_type): f"/similarity_info/{repr(fp_type)}/fps" for fp_type in fingerprint}

        # input and output queue
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()
        # define number of workers (keep 1 reader and 1 writer)
        if njobs == -1:
            n_cpus = multiprocessing.cpu_count() - 2  # number of threads (logical cores)
        else:
            n_cpus = njobs - 1
        processes = []
        # Start reader
        reader = multiprocessing.Process(target=_reader_process, args=(self.sd_file, n_cpus,
                                                                       total, False,
                                                                       input_queue))
        processes.append(reader)
        reader.start()
        # Start writer
        writer = multiprocessing.Process(target=_writer_process, args=(self.h5_filename, output_queue,
                                                                       table_paths, total, progress))
        writer.start()
        # Start workers
        for i in range(n_cpus):
            job = multiprocessing.Process(target=_worker_process, args=(fp_types, input_queue, output_queue))
            processes.append(job)
            processes[-1].start()
        # Joining workers
        while len(processes):
            processes[0].join(10)
            if not processes[0].is_alive():
                del processes[0]
        output_queue.put('STOP')
        writer.join()
        input_queue.close()
        input_queue.join_thread()
        output_queue.close()
        output_queue.join_thread()
        # sort by popcounts
        sort_db_file(self.h5_filename, verbose=progress)

    @property
    def available_fingerprints(self):
        if hasattr(self, '_avail_fp'):
            return self._avail_fp
        self._avail_fp = {}
        with tb.open_file(self.h5_filename, mode="r") as h5file:
            for simfp_group in h5file.walk_groups('/similarity_info/'):
                if len(simfp_group._v_name):
                    fp_table = h5file.get_node(simfp_group, 'fps', classname='Table')
                    fp_type = fp_table.attrs.fp_type
                    fp_params = json.loads(fp_table.attrs.fp_params)
                    self._avail_fp[fp_table.attrs.fp_id] = get_fp_from_name(fp_type, **fp_params)
        return self._avail_fp

    def get_substructure_lib(self):
        if not os.path.isfile(self.h5_filename):
            raise ValueError('file must be created first')
        with tb.open_file(self.h5_filename, mode="r") as h5file:
            padding = h5file.root.substructure_info.substruct_lib.attrs.padding
            data = h5file.root.substructure_info.substruct_lib.read()
        with BytesIO(data.tobytes('C')[:-padding]) as stream:
            lib = SubstructureLibrary(self.h5_filename)
            lib.InitFromStream(stream)
        return lib

    def get_similarity_lib(self, fp_signature: Optional[str] = None, cuda: bool = False):
        """Obtain a similarity engine for the desired fingerprint.

        :param fp_signature: Signature of the desired fingerprint
        :param cuda: whether to run searches on the GPU
        """
        if not os.path.isfile(self.h5_filename):
            raise ValueError('file must be created first')
        _ = self.available_fingerprints  # initialize self._avail_fp
        if fp_signature not in [*self._avail_fp.keys(), None]:
            raise ValueError(f'fingerprint not available, choose one of {self._avail_fp.keys()}')
        elif fp_signature is None:
            fp_signature = list(self._avail_fp.keys())[0]
        if cuda:
            return FPSubSim2CudaEngine(self.h5_filename, fp_signature)
        return FPSubSim2Engine(self.h5_filename, fp_signature)

    def add_fingerprint(self, fingerprint: Fingerprint,
                        papyrus_sd_file: str,
                        progress: bool = True,
                        total: Optional[int] = None):
        """Add a similarity fingerprint to the FPSubSim2 database.

        :param fingerprint: Fingerprint to be added
        :param papyrus_sd_file: papyrus sd file containing chemical structures
        :param progress: whether progress should be shown
        :param total: number of molecules for progress display
        """
        signature = str(fingerprint)
        available_fps = [*self.available_fingerprints.keys()]
        if signature in available_fps:
            print(f'fingerprint f{signature} is already available')
            return
        backend = PyTablesMultiFpStorageBackend(self.h5_filename, available_fps[0])
        backend.change_fp_for_append(fingerprint)
        backend.append_fps(MolSupplier(source=papyrus_sd_file), total=total, progress=progress)

    def add_molecules(self, papyrus_sd_file: str, progress: bool = True, total: Optional[int] = None):
        """Add molecules to the FPSubSim2 database.

        :param papyrus_sd_file: papyrus sd file containing new chemical structures
        :param progress: whether progress should be shown
        :param total: number of molecules for progress display
        """
        for signature, fingerprint in self.available_fingerprints:
            backend = PyTablesMultiFpStorageBackend(self.h5_filename, signature)
            backend.append_fps(MolSupplier(source=papyrus_sd_file), total=total, progress=progress, sort=False)
        substruct_lib = self.get_substructure_lib()
        for rdmol in MolSupplier(source=papyrus_sd_file, total=total, progress=progress):
            if rdmol is not None:
                substruct_lib.AddMol(rdmol)
        # serialize substruct lib and pad
        lib_bytes = substruct_lib.Serialize()
        remainder = len(lib_bytes) % 8
        padding = 8 - remainder if remainder else 0  # int64 are 8 bytes
        if padding:
            lib_bytes += b'\x00' * padding
        lib_ints = np.frombuffer(lib_bytes, dtype=np.int64)
        # save into h5
        with tb.open_file(self.h5_filename, mode="a") as h5file:
            # Remove previous lib
            h5file.remove_node(h5file.root.substructure_info.substruct_lib)
            h5file.create_earray(h5file.root.substructure_info, 'substruct_lib',
                                 tb.UInt64Atom(), (0,), 'Substructure search library')
            h5file.root.substructure_info.substruct_lib.attrs.padding = padding
            h5file.root.substructure_info.substruct_lib.append(lib_ints)
        sort_db_file(self.h5_filename, verbose=progress)


def _reader_process(sd_file, n_workers, total, progress, input_queue):
    with MolSupplier(source=sd_file, total=total, show_progress=progress, start_id=1) as supplier:
        count = 0
        for mol_id, rdmol in supplier:
            input_queue.put((mol_id, rdmol, rdmol.GetPropsAsDict()))
            # Allow the queue to get emptied periodically
            count += 1
            if count > BATCH_WRITE_SIZE * n_workers * 1.5:
                while input_queue.qsize() > BATCH_WRITE_SIZE:
                    time.sleep(10)
                count = 0
    for _ in range(n_workers):
        input_queue.put('END')


def _writer_process(h5_filename, output_queue, table_paths, total, progress):
    lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
    pbar = tqdm(total=total, smoothing=0.0) if progress else {}
    mappings_insert = []
    similarity_insert = defaultdict(list)
    with tb.open_file(h5_filename, mode="r+") as h5file:
        while True:
            data = output_queue.get()
            if data == 'STOP':
                # flush remnants of data
                h5file.root.mol_mappings.append(mappings_insert)
                for fp_id, fp_insert in similarity_insert.items():
                    h5file.get_node(table_paths[fp_id]).append(fp_insert)
                # serialize substructure lib and pad
                lib_bytes = lib.Serialize()
                remainder = len(lib_bytes) % 8
                padding = 8 - remainder if remainder else 0  # int64 are 8 bytes
                if padding:
                    lib_bytes += b'\x00' * padding
                lib_ints = np.frombuffer(lib_bytes, dtype=np.int64)
                h5file.root.substructure_info.substruct_lib.attrs.padding = padding
                h5file.root.substructure_info.substruct_lib.append(lib_ints)
                # create index so tables can be sorted
                for fp_table_path in table_paths.values():
                    h5file.get_node(fp_table_path).cols.popcnt.create_index(kind="full")
                h5file.root.mol_mappings.cols.idnumber.create_index(kind="full")
                break
            if data[0] == 'mappings':
                mappings_insert.append(data[1])
                pbar.update()
            elif data[0] == 'substructure':
                lib.AddMol(data[1])
                del data
            elif data[0] == 'similarity':
                fp_id, fp = data[1], data[2]
                similarity_insert[fp_id].append(fp)
            # insert data
            if len(mappings_insert) > BATCH_WRITE_SIZE:
                h5file.root.mol_mappings.append(mappings_insert)
                h5file.root.mol_mappings.flush()
                mappings_insert = []
            if any(len(x) > BATCH_WRITE_SIZE for x in similarity_insert.values()):
                for fp_id, fp_insert in similarity_insert.items():
                    h5file.get_node(table_paths[fp_id]).append(fp_insert)
                    h5file.get_node(table_paths[fp_id]).flush()
                similarity_insert = defaultdict(list)
    # ensure index in mol_mappings
    with tb.open_file(h5_filename, mode="r+") as h5file:
        h5file.root.mol_mappings.cols.idnumber.reindex()
    return


def _worker_process(fp_types, input_queue, output_queue):
    while True:
        # while output_queue.qsize() > BATCH_WRITE_SIZE * n_workers / 2:
        #     time.sleep(0.5)
        data = input_queue.get()
        if data == 'END':
            # pass end signal to writing process
            break
        mol_id, rdmol, props = data
        # put the molecule for the writer to handle substructure
        output_queue.put(('substructure', rdmol))
        # handle mappings
        connectivity = props.get('connectivity', '')
        inchikey = props.get('InChIKey', '')
        if not inchikey and connectivity:
            connectivity = inchikey.split('-')[0]
        output_queue.put(('mappings', (mol_id, connectivity, inchikey)))
        for fp_type, fp_params in fp_types:
            fper = fp_type(**fp_params)
            # generate fingerprint
            fp = fper.get(rdmol)
            output_queue.put(('similarity', repr(fper), (mol_id, *fp)))


def sort_db_file(filename: str, verbose: bool = False) -> None:
    """Sorts the FPs db file."""
    if verbose:
        print('Optimizing FPSubSim2 file.')
    # rename not sorted filename
    tmp_filename = filename + "_tmp"
    if os.path.isfile(tmp_filename):
        os.remove(tmp_filename)
    os.rename(filename, tmp_filename)
    filters = tb.Filters(complib="blosc", complevel=1, shuffle=True, bitshuffle=True)
    stats = {
        "groups": 0,
        "leaves": 0,
        "links": 0,
        "bytes": 0,
        "hardlinks": 0,
        }

    # copy sorted fps and config to a new file
    with tb.open_file(tmp_filename, mode="r") as fp_file:
        with tb.open_file(filename, mode="w") as sorted_fp_file:
            # group to hold similarity tables
            siminfo_group = sorted_fp_file.create_group(sorted_fp_file.root, "similarity_info",
                                                        "Infos for similarity search")
            simfp_groups = list(fp_file.walk_groups('/similarity_info/'))
            i = 0
            for simfp_group in simfp_groups:
                if len(simfp_group._v_name):
                    dst_group = simfp_group._f_copy(siminfo_group, recursive=False, filters=filters, stats=stats)
                    # progress bar
                    if verbose:
                        pbar = tqdm(list(fp_file.iter_nodes(simfp_group, classname='Table')),
                                    desc=f'Optimizing tables of group ({i}/{len(simfp_groups)})',
                                    leave=False)
                    else:
                        pbar = fp_file.iter_nodes(simfp_group, classname='Table')
                    for fp_table in pbar:
                        # create a sorted copy of the fps table
                        dst_fp_table = fp_table.copy(
                            dst_group,
                            fp_table.name,
                            filters=filters,
                            copyuserattrs=True,
                            overwrite=True,
                            stats=stats,
                            start=None,
                            stop=None,
                            step=None,
                            chunkshape="auto",
                            sortby="popcnt",
                            check_CSI=True,
                            propindexes=True,
                        )

                        # update count ranges
                        popcnt_bins = calc_popcnt_bins_pytables(dst_fp_table, fp_table.attrs.length)
                        popcounts = sorted_fp_file.create_vlarray(dst_group, 'popcounts', tb.ObjectAtom(),
                                                                  f'Popcounts of {dst_group._v_name}')
                        for x in popcnt_bins:
                            popcounts.append(x)
            # add other tables
            if verbose:
                print('Optimizing remaining groups and arrays.')
            for node in fp_file.iter_nodes(fp_file.root):
                if isinstance(node, tb.group.Group):
                    if isinstance(node, tb.group.RootGroup) or 'similarity_info' in str(node):
                        continue
                    _ = node._f_copy(sorted_fp_file.root, node._v_name,
                                     overwrite=True, recursive=True,
                                     filters=filters, stats=stats)
                else:
                    _ = node.copy(sorted_fp_file.root, node._v_name, overwrite=True, stats=stats)
    # remove unsorted file
    if verbose:
        print('Cleaning up temporary files.')
    os.remove(tmp_filename)


class PyTablesMultiFpStorageBackend(BaseStorageBackend):
    def __init__(self, fp_filename: str, fp_signature: str, in_memory_fps: bool = True, fps_sort: bool = False) -> None:
        super(PyTablesMultiFpStorageBackend, self).__init__(fp_filename)
        self.name = "pytables"
        # Get table signatures
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            self._fp_table_mappings = {}
            for simfp_group in fp_file.walk_groups('/similarity_info/'):
                if len(simfp_group._v_name):
                    fp_table = fp_file.get_node(simfp_group, 'fps', classname='Table')
                    self._fp_table_mappings[fp_table.attrs.fp_id] = [f'/similarity_info/{simfp_group._v_name}'
                                                                     '/fps',
                                                                     f'/similarity_info/{simfp_group._v_name}'
                                                                     '/popcounts']
        if fp_signature not in self._fp_table_mappings.keys():
            raise ValueError(f'fingerprint not available, must be one of {", ".join(self._fp_table_mappings.keys())}')
        self._current_fp = fp_signature
        self._current_fp_path = self._fp_table_mappings[fp_signature][0]
        self._current_popcounts_path = self._fp_table_mappings[fp_signature][1]
        self.fp_type, self.fp_params, self.rdkit_ver = self.read_parameters()
        self._fp_func = get_fp_from_name(self.fp_type, **self.fp_params)
        if in_memory_fps:
            self.load_fps(in_memory_fps, fps_sort)
        self.load_popcnt_bins(fps_sort)
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            self.chunk_size = fp_file.get_node(self._current_fp_path).chunkshape[0] * 120

    def read_parameters(self) -> Tuple[str, Dict[str, Dict[str, dict]], str]:
        """Reads fingerprint parameters for the current fingerprint."""
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            rdkit_ver = fp_file.root.config[0]
            fp_table = fp_file.get_node(self._current_fp_path)
            fp_type = fp_table.attrs.fp_type
            fp_params = json.loads(fp_table.attrs.fp_params)
        return fp_type, fp_params, rdkit_ver

    def get_fps_chunk(self, chunk_range: Tuple[int, int]) -> np.asarray:
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fps = fp_file.get_node(self._current_fp_path)[slice(*chunk_range)]
        return fps

    def load_popcnt_bins(self, fps_sort: bool) -> None:
        if fps_sort:
            popcnt_bins = self.calc_popcnt_bins(self.fps)
        else:
            with tb.open_file(self.fp_filename, mode="r") as fp_file:
                popcnt_bins = fp_file.get_node(self._current_popcounts_path).read()
        self.popcnt_bins = popcnt_bins

    def load_fps(self, in_memory_fps, fps_sort) -> None:
        """Loads FP db file into memory for the current fingerprint.
        Parameters
        ----------
        in_memory_fps : bool
            Whether if the FPs should be loaded into memory or not.
        fps_sort: bool
            Whether if the FPs should be sorted or not.
        Returns
        -------
        fps: numpy array
            Numpy array with the fingerprints.
        """
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fps = fp_file.get_node(self._current_fp_path)[:]
            # files should be sorted but if the file is updated without sorting it
            # can be also in memory sorted
            if fps_sort:
                fps.sort(order="popcnt")
        num_fields = len(fps[0])
        fps = fps.view("<u8")
        fps = fps.reshape(int(fps.size / num_fields), num_fields)
        self.fps = fps

    def delete_fps(self, ids_list: List[int]) -> None:
        """Delete FPs given a list of ids for the current fingerprint.
        Parameters
        ----------
        ids_list : list
            ids to delete.
        Returns
        -------
        None
        """
        with tb.open_file(self.fp_filename, mode="a") as fp_file:
            fps_table = fp_file.get_node(self._current_fp_path)
            for fp_id in ids_list:
                to_delete = [
                    row.nrow
                    for row in fps_table.where("fp_id == {}".format(str(fp_id)))
                ]
                fps_table.remove_row(to_delete[0])

    def append_fps(self, supplier: MolSupplier, progress: bool = True,
                   total: Optional[int] = None, sort: bool = True) -> None:
        """Appends FPs to the file for the fingerprint currently selected."""
        with tb.open_file(self.fp_filename, mode="a") as fp_file:
            fps_table = fp_file.get_node(self._current_fp_path)
            fps = []
            supplier.set_start_progress_total(max((row['fp_id'] for row in fps_table.iterrows()), default=1),
                                              progress, total)
            for mol_id, rdmol in supplier:
                if not rdmol:
                    continue
                fp = self._fp_func.get(rdmol)
                fps.append((mol_id, *fp))
                if len(fps) == BATCH_WRITE_SIZE:
                    fps_table.append(fps)
                    fps = []
            # append last batch < 32k
            if fps:
                fps_table.append(fps)
        if sort:
            sort_db_file(self.fp_filename, verbose=progress)

    def change_fp_for_append(self, fingerprint: Fingerprint):
        """Create an empty table and change the fingerprint to be used for appending."""
        self._current_fp = str(fingerprint)
        # Determine schema
        particle = create_schema(fingerprint.length)
        filters = tb.Filters()
        # Create table
        with tb.open_file(self.fp_filename, mode="a") as fp_file:
            # Group for new fingerprint
            fp_file.create_table('/similarity_info/', )
            fp_group = fp_file.create_group('/similarity_info/', self._current_fp, f'Similarity {self._current_fp}')
            # New table
            particle = create_schema(fingerprint.length)
            fp_table = fp_file.create_table(fp_group, 'fps', particle, 'Similarity FPs', expectedrows=1300000,
                                            filters=filters)
            # New attributes
            fp_table.attrs.fp_type = fingerprint.name
            fp_table.attrs.fp_id = self._current_fp
            fp_table.attrs.length = fingerprint.length
            fp_table.attrs.fp_params = json.dumps(fingerprint.params)
            # New Popcounts
            popcounts = fp_file.create_vlarray(fp_group, 'popcounts', tb.ObjectAtom(),
                                               f'Popcounts of {fp_group._v_name}')
        self._current_fp_path = f'/similarity_info/{fp_group._v_name}/fps'
        self._current_popcounts_path = f'/similarity_info/{fp_group._v_name}/popcounts'
        self.fp_type, self.fp_params, self.rdkit_ver = self.read_parameters()
        self._fp_func = get_fp_from_name(self.fp_type, **self.fp_params)
        print('Empty table created, make sure to call "append_fps" to populate it!')


class BaseMultiFpEngine(BaseEngine, ABC):

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
        if storage_backend == "pytables":
            self.storage = PyTablesMultiFpStorageBackend(
                fp_filename, fp_signature, in_memory_fps=in_memory_fps, fps_sort=fps_sort
            )

    def load_query(self, query_string: str) -> np.ndarray:
        """Loads the query molecule from SMILES, molblock or InChI.
        Parameters
        ----------
        query_string : str
            SMILES, InChi or molblock.
        Returns
        -------
        query : numpy array
            Numpy array query molecule.
        """
        rdmol = load_molecule(query_string)
        if rdmol is None:
            raise ValueError('molecule could not be parsed')
        fp = get_fp_from_name(self.fp_type, **self.fp_params).get(rdmol)
        return np.array((0, *fp), dtype=np.uint64)

    def _get_mapping(self, ids: Union[List[int], int]):
        """Get the Papyrus identifiers corresponding to the given indices"""
        if not isinstance(ids, list):
            ids = [ids]
        if not len(ids):
            raise ValueError('indices must be supplied')
        for id in ids:
            if int(id) != id:
                raise ValueError('indices must be integers')
        with tb.open_file(self.fp_filename) as fp_file:
            if max(ids) > max(fp_file.get_node(self.storage._current_fp_path).cols.fp_id):
                raise ValueError(f'index not in database: {max(ids)}')
            # Get data fields from mol_mappings table
            mappings_table = fp_file.root.mol_mappings
            colnames = mappings_table.cols._v_colnames
            data = []
            for id in ids:
                pointer = mappings_table.where(f"idnumber == {id}")
                try:
                    data.append(next(pointer).fetch_all_fields())
                except StopIteration:
                    raise ValueError(f'could not find index {id}')
        return pd.DataFrame.from_records(data, columns=colnames)


class FPSubSim2Engine(BaseMultiFpEngine, FPSim2Engine):
    """FPSubSim2 class to run fast CPU similarity searches."""

    def __init__(
        self,
        fp_filename: str,
        fp_signature: str,
        in_memory_fps: bool = True,
        fps_sort: bool = False,
        storage_backend: str = "pytables",
    ) -> None:
        """FPSubSim2 class to run fast CPU similarity searches.

        :param fp_filename : Fingerprints database file path.
        :param in_memory_fps: Whether if the FPs should be loaded into memory or not.
        :param fps_sort: Whether if the FPs should be sorted by popcnt after being loaded into memory or not.
        :param storage_backend: Storage backend to use (only pytables available at the moment).
        """
        super(FPSubSim2Engine, self).__init__(
            fp_filename=fp_filename,
            fp_signature=fp_signature,
            storage_backend=storage_backend,
            in_memory_fps=in_memory_fps,
            fps_sort=fps_sort,
        )
        self.empty_sim = np.ndarray((0,), dtype=[("mol_id", "<u4"), ("coeff", "<f4")])
        self.empty_subs = np.ndarray((0,), dtype="<u4")

    def similarity(self, query_string: str, threshold: float, n_workers: int = 1) -> pd.DataFrame:
        """Perform in-memory Tanimoto similarity search.

        :param query_string:
        :param threshold:
        :param n_workers:
        :return:
        """
        data = list(zip(*FPSim2Engine.similarity(self, query_string, threshold, n_workers)))
        if not len(data):
            return pd.DataFrame([], columns=['idnumber', 'connectivity', 'InChIKey',
                                             f'Tanimoto > {threshold} ({self.storage._current_fp})'])
        ids, similarities = data
        ids, similarities = list(ids), list(similarities)
        data = self._get_mapping(ids)
        data[f'Tanimoto > {threshold} ({self.storage._current_fp})'] = similarities
        # Decode byte columns
        for col, dtype in data.dtypes.items():
            if dtype == object:
                data[col] = data[col].apply(lambda x: x.decode('utf-8'))
        return data

    def on_disk_similarity(self, query_string: str, threshold: float, n_workers: int = 1, chunk_size: int = 0):
        """Perform Tanimoto similarity search on disk.

        :param query_string:
        :param threshold:
        :param n_workers:
        :param chunk_size:
        :return:
        """
        data = list(zip(*FPSim2Engine.on_disk_similarity(self, query_string, threshold, n_workers, chunk_size)))
        if not len(data):
            return pd.DataFrame([], columns=['idnumber', 'connectivity', 'InChIKey',
                                             f'Tanimoto > {threshold} ({self.storage._current_fp})'])
        ids, similarities = data
        ids, similarities = list(ids), list(similarities)
        data = self._get_mapping(ids)
        data[f'Tanimoto > {threshold} ({self.storage._current_fp})'] = similarities
        # Decode byte columns
        for col, dtype in data.dtypes.items():
            if dtype == object:
                data[col] = data[col].apply(lambda x: x.decode('utf-8'))
        return data

    def tversky(self, query_string: str, threshold: float, a: float, b: float, n_workers: int = 1):
        """Perform in-memory Tversky similarity search.

        :param query_string:
        :param threshold:
        :param a:
        :param b:
        :param n_workers:
        :return:
        """
        data = list(zip(*FPSim2Engine.tversky(self, query_string, threshold, a, b, n_workers)))
        if not len(data):
            return pd.DataFrame([], columns=['idnumber', 'connectivity', 'InChIKey',
                                             f'Tversky > {threshold} ({self.storage._current_fp})'])
        ids, similarities = data
        ids, similarities = list(ids), list(similarities)
        data = self._get_mapping(ids)
        data[f'Tanimoto > {threshold} ({self.storage._current_fp})'] = similarities
        # Decode byte columns
        for col, dtype in data.dtypes.items():
            if dtype == object:
                data[col] = data[col].apply(lambda x: x.decode('utf-8'))
        return data

    def on_disk_tversky(self, query_string: str, threshold: float,
                        a: float, b: float,
                        n_workers: int = 1, chunk_size: int = None):
        """Perform Tversky similarity search on disk.

        :param query_string:
        :param threshold:
        :param a:
        :param b:
        :param n_workers:
        :param chunk_size:
        :return:
        """
        data = list(zip(*FPSim2Engine.on_disk_tversky(self, query_string, threshold, a, b, n_workers, chunk_size)))
        if not len(data):
            return pd.DataFrame([], columns=['idnumber', 'connectivity', 'InChIKey',
                                             f'Tversky > {threshold} ({self.storage._current_fp})'])
        ids, similarities = data
        ids, similarities = list(ids), list(similarities)
        data = self._get_mapping(ids)
        data[f'Tanimoto > {threshold} ({self.storage._current_fp})'] = similarities
        # Decode byte columns
        for col, dtype in data.dtypes.items():
            if dtype == object:
                data[col] = data[col].apply(lambda x: x.decode('utf-8'))
        return data

    def substructure(self, query_string: str, n_workers: int = 1):
        raise ValueError('use FPSubSim2 substructure library granting subgraph isomorphism')

    def on_disk_substructure(self, query_string: str, n_workers: int = 1, chunk_size: int = None):
        raise ValueError('use FPSubSim2 substructure library granting subgraph isomorphism')


class FPSubSim2CudaEngine(BaseMultiFpEngine, FPSim2CudaEngine):
    """FPSubSim2 class to run fast GPU similarity searches."""

    def __init__(
        self,
        fp_filename: str,
        fp_signature: str,
        storage_backend: str = "pytables",
        kernel: str = 'raw'
    ) -> None:
        """FPSubSim2 class to run fast CPU similarity searches.

        :param fp_filename : Fingerprints database file path.
        :param in_memory_fps: Whether if the FPs should be loaded into memory or not.
        :param fps_sort: Whether if the FPs should be sorted by popcnt after being loaded into memory or not.
        :param storage_backend: Storage backend to use (only pytables available at the moment).
        """
        if isinstance(cupy, ImportError):
            raise ImportError('Some required dependencies are missing:\n\tcupy')
        super(FPSubSim2CudaEngine, self).__init__(
            fp_filename=fp_filename,
            fp_signature=fp_signature,
            storage_backend=storage_backend,
            in_memory_fps=True,
            fps_sort=False,
        )
        if kernel not in ['raw', 'element_wise']:
            raise ValueError("only supports 'raw' and 'element_wise' kernels")
        self.kernel = kernel
        if kernel == "raw":
            # copy all the stuff to the GPU
            self.cuda_db = cupy.asarray(self.fps[:, 1:-1])
            self.cuda_ids = cupy.asarray(self.fps[:, 0])
            self.cuda_popcnts = cupy.asarray(self.fps[:, -1])
            self.cupy_kernel = cupy.RawKernel(
                self.raw_kernel.format(block=self.cuda_db.shape[1]),
                name="taniRAW",
                options=("-std=c++14",),
            )

        elif self.kernel == "element_wise":
            # copy the database to the GPU
            self.cuda_db = cupy.asarray(self.fps)
            self.cupy_kernel = cupy.ElementwiseKernel(
                in_params="raw T db, raw U query, uint64 in_width, float32 threshold",
                out_params="raw V out",
                operation=self.ew_kernel,
                name="taniEW",
                options=("-std=c++14",),
                reduce_dims=False,
            )

    def similarity(self, query_string: str, threshold: float) -> pd.DataFrame:
        """Tanimoto similarity search."""
        data = list(zip(*FPSim2CudaEngine.similarity(self, query_string, threshold)))
        if not len(data):
            return pd.DataFrame([], columns=['idnumber', 'connectivity', 'InChIKey',
                                             f'Tanimoto > {threshold} ({self.storage._current_fp})'])
        ids, similarities = data
        ids, similarities = list(ids), list(similarities)
        data = self._get_mapping(ids)
        data[f'Tanimoto > {threshold} ({self.storage._current_fp})'] = similarities
        return data


class SubstructureLibrary(SubstructLibrary):
    def __init__(self, fp_file_name):
        """Extenstion of RDKIT's rdSubstructLibrary to support mappings.

        :param fp_file_name: file containing the molecular mappings of the substructure library
        """
        super(SubstructureLibrary, self).__init__()
        self.lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
        self.fp_filename = fp_file_name

    def _get_mapping(self, ids: Union[List[int], int]):
        """Get the Papyrus identifiers corresponding to the given indices"""
        if not isinstance(ids, list):
            ids = [ids]
        if not len(ids):
            raise ValueError('indices must be supplied')
        for id in ids:
            if int(id) != id:
                raise ValueError('indices must be integers')
        with tb.open_file(self.fp_filename) as fp_file:
            if max(ids) > max(fp_file.root.mol_mappings.cols.idnumber):
                raise ValueError(f'index not in database: {max(ids)}')
            # Get data fields from mol_mappings table
            mappings_table = fp_file.root.mol_mappings
            colnames = mappings_table.cols._v_colnames
            data = []
            for id in ids:
                pointer = mappings_table.where(f"idnumber == {id}")
                try:
                    data.append(next(pointer).fetch_all_fields())
                except StopIteration:
                    raise ValueError(f'could not find index {id}')
        data = pd.DataFrame.from_records(data, columns=colnames)
        # Decode byte columns
        for col, dtype in data.dtypes.items():
            if dtype == object:
                data[col] = data[col].apply(lambda x: x.decode('utf-8'))
        return data

    def GetMatches(self, query: Union[str, Chem.Mol], recursionPossible: bool = True, useChirality: bool = True,
                   useQueryQueryMatches: bool = False, numThreads: int = -1, maxResults: int = -1):
        if isinstance(query, str):
            query = load_molecule(query)
        ids = list(super(SubstructureLibrary, self).GetMatches(query=query,
                                                               recursionPossible=recursionPossible,
                                                               useChirality=useChirality,
                                                               useQueryQueryMatches=useQueryQueryMatches,
                                                               numThreads=numThreads,
                                                               maxResults=maxResults))
        return self._get_mapping(ids)

    def substructure(self, query: Union[str, Chem.Mol]):
        return self.GetMatches(query)
