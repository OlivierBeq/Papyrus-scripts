# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.subsim_search.FPSubSim2._parallel_create.

multiprocessing.Process/Queue are mocked entirely: no real worker processes
are spawned, no .h5 file or SD file is touched. FPSubSim2.__init__ is
bypassed (via __new__) in most tests to avoid depending on optional deps
(tables, FPSim2) being installed. Tests simulating an absent dep must
explicitly patch its HAS_* flag rather than rely on ambient absence, since
these must pass regardless of what's actually installed.
"""

import itertools
import queue
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import polars as pl
from rdkit import Chem

from src.papyrus_scripts import fingerprint as fp
from src.papyrus_scripts import subsim_search as ss
from src.papyrus_scripts.fingerprint import Fingerprint, MorganFingerprint
from src.papyrus_scripts.subsim_search import (
    HAS_FPSIM2,
    HAS_TABLES,
    FPSubSim2,
    _build_result_df,
    _check_optional_deps,
    _decode_bytes_df,
    _derive_connectivity,
    _fp_table_path,
    _MappingMixin,
    _pad_to_int64,
    _padding_for,
    _serialize_substruct_lib,
    _validate_fingerprints,
)


def make_engine():
    engine = FPSubSim2.__new__(FPSubSim2)
    engine.sd_file = 'fake_sd_file.sdf'
    engine.h5_filename = 'fake_output.h5'
    engine.version = None
    engine.is3d = False
    return engine


class TestCheckOptionalDeps(unittest.TestCase):

    def test_raises_when_both_missing(self):
        with patch.object(ss, 'HAS_TABLES', False), patch.object(ss, 'HAS_FPSIM2', False):
            with self.assertRaises(ImportError) as ctx:
                _check_optional_deps()
        self.assertIn('tables', str(ctx.exception))
        self.assertIn('FPSim2', str(ctx.exception))

    def test_raises_listing_only_missing_one(self):
        with patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', False):
            with self.assertRaises(ImportError) as ctx:
                _check_optional_deps()
        self.assertNotIn('tables', str(ctx.exception))
        self.assertIn('FPSim2', str(ctx.exception))

    def test_does_not_raise_when_both_available(self):
        with patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True):
            _check_optional_deps()  # must not raise


class TestValidateFingerprints(unittest.TestCase):

    def test_none_returns_all_derived(self):
        # Fingerprint.derived() includes OBFingerprint subclasses that need
        # openbabel/FPSim2 (not installed here) - stub it to instantiable
        # classes only, since _validate_fingerprints itself doesn't care.
        with patch.object(Fingerprint, 'derived', return_value=[MorganFingerprint]):
            result = _validate_fingerprints(None)
        self.assertTrue(all(isinstance(fp, Fingerprint) for fp in result))
        self.assertGreater(len(result), 0)

    def test_single_fingerprint_wrapped_in_list(self):
        fp = MorganFingerprint()
        result = _validate_fingerprints(fp)
        self.assertEqual(result, [fp])

    def test_list_passthrough(self):
        fps = [MorganFingerprint(), MorganFingerprint(nBits=1024)]
        self.assertEqual(_validate_fingerprints(fps), fps)

    def test_invalid_element_raises(self):
        with self.assertRaises(ValueError):
            _validate_fingerprints([MorganFingerprint(), 'not-a-fingerprint'])


class TestFpTablePath(unittest.TestCase):

    def test_format(self):
        fp = MorganFingerprint()
        self.assertEqual(_fp_table_path(fp), f'/similarity_info/{repr(fp)}/fps')


class TestDeriveConnectivity(unittest.TestCase):

    def test_uses_provided_inchikey_and_connectivity(self):
        mol = Chem.MolFromSmiles('CCO')
        connectivity, inchikey = _derive_connectivity(
            {'InChIKey': 'AAAAAAAAAAAAAA-BBBBBBBBBB-C', 'connectivity': 'AAAAAAAAAAAAAA'}, mol,
        )
        self.assertEqual(connectivity, 'AAAAAAAAAAAAAA')
        self.assertEqual(inchikey, 'AAAAAAAAAAAAAA-BBBBBBBBBB-C')

    def test_derives_connectivity_from_inchikey_when_missing(self):
        mol = Chem.MolFromSmiles('CCO')
        connectivity, inchikey = _derive_connectivity({'InChIKey': 'AAAAAAAAAAAAAA-BBBBBBBBBB-C'}, mol)
        self.assertEqual(connectivity, 'AAAAAAAAAAAAAA')

    def test_computes_inchikey_from_molecule_when_missing(self):
        mol = Chem.MolFromSmiles('CCO')
        connectivity, inchikey = _derive_connectivity({}, mol)
        self.assertTrue(inchikey)
        self.assertEqual(connectivity, inchikey.split('-')[0])


class TestDecodeBytesDf(unittest.TestCase):

    def test_decodes_object_column_with_bytes(self):
        df = pl.DataFrame({'a': pl.Series([b'x', b'y'], dtype=pl.Object), 'b': [1, 2]})
        out = _decode_bytes_df(df)
        self.assertEqual(out['a'].to_list(), ['x', 'y'])
        self.assertEqual(out['b'].to_list(), [1, 2])

    def test_non_object_columns_untouched(self):
        df = pl.DataFrame({'a': ['x', 'y'], 'b': [1, 2]})
        out = _decode_bytes_df(df)
        self.assertEqual(out['a'].to_list(), ['x', 'y'])

    def test_decodes_binary_column_from_pytables_fixed_length_strings(self):
        # PyTables 'S' columns surface as pl.Binary, not pl.Object.
        df = pl.DataFrame({'a': pl.Series([b'x', b'y'], dtype=pl.Binary), 'b': [1, 2]})
        out = _decode_bytes_df(df)
        self.assertEqual(out.schema['a'], pl.Utf8)
        self.assertEqual(out['a'].to_list(), ['x', 'y'])


class TestBuildResultDf(unittest.TestCase):

    def test_empty_results_returns_empty_schema(self):
        df = _build_result_df([], get_mapping=MagicMock(), score_col='score')
        self.assertEqual(df.height, 0)
        self.assertEqual(set(df.columns), {'idnumber', 'connectivity', 'InChIKey', 'score'})

    def test_non_empty_results_adds_score_column(self):
        mapping_df = pl.DataFrame({
            'idnumber': [1, 2], 'connectivity': ['a', 'b'], 'InChIKey': ['x', 'y'],
        })
        get_mapping = MagicMock(return_value=mapping_df)
        raw = [(1, 0.9), (2, 0.5)]
        df = _build_result_df(raw, get_mapping, 'score')
        get_mapping.assert_called_once_with([1, 2])
        self.assertEqual(df['score'].to_list(), [0.9, 0.5])


class TestPaddingHelpers(unittest.TestCase):

    def test_padding_for_exact_multiple_is_zero(self):
        self.assertEqual(_padding_for(b'12345678'), 0)

    def test_padding_for_remainder(self):
        self.assertEqual(_padding_for(b'123'), 5)

    def test_pad_to_int64_pads(self):
        self.assertEqual(_pad_to_int64(b'123'), b'123' + b'\x00' * 5)

    def test_pad_to_int64_noop_when_aligned(self):
        self.assertEqual(_pad_to_int64(b'12345678'), b'12345678')

    def test_serialize_substruct_lib_roundtrip(self):
        from rdkit.Chem.rdSubstructLibrary import CachedMolHolder, PatternHolder, SubstructLibrary
        lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
        lib.AddMol(Chem.MolFromSmiles('CCO'))
        data = _serialize_substruct_lib(lib)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)


class _FakeGroup:
    """Stand-in for tb.group.Group - real class needed since sort_db_file
    uses isinstance() against tb.group.Group/RootGroup, which a plain
    MagicMock attribute can't satisfy.
    """

    def __init__(self, name):
        self._v_name = name

    def __str__(self):
        return self._v_name


class _FakeRootGroup(_FakeGroup):
    pass


class TestSortDbFile(unittest.TestCase):

    def _run(self, progress):
        fp_table = MagicMock()
        fp_table.attrs.length = 2048
        dst_fp_table = MagicMock()
        fp_table.copy.return_value = dst_fp_table

        simfp_group = _FakeGroup('Morgan_2048bits_x')
        skipped_group = _FakeGroup('')  # _v_name falsy -> skipped by sort_db_file itself

        root_node = _FakeRootGroup('root')
        siminfo_node = _FakeGroup('similarity_info')
        other_group_node = _FakeGroup('other_group')
        array_node = MagicMock()
        array_node._v_name = 'some_array'

        src = MagicMock()
        src.walk_groups.return_value = [skipped_group, simfp_group]
        src.iter_nodes.side_effect = lambda node=None, classname=None: (
            [fp_table] if classname == 'Table' else [root_node, siminfo_node, other_group_node, array_node]
        )

        dst = MagicMock()
        simfp_group_copy = MagicMock()
        # _f_copy is called on the (non-dunder-clashing) group instances too;
        # attach as a bound-like attribute since _FakeGroup doesn't define it.
        simfp_group._f_copy = MagicMock(return_value=simfp_group_copy)
        skipped_group._f_copy = MagicMock()
        other_group_node._f_copy = MagicMock()

        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'calc_popcnt_bins_pytables', create=True, return_value=[b'\x00']),
        ):
            mock_tb.group.Group = _FakeGroup
            mock_tb.group.RootGroup = _FakeRootGroup
            mock_tb.open_file.side_effect = [
                MagicMock(__enter__=MagicMock(return_value=src), __exit__=MagicMock(return_value=False)),
                MagicMock(__enter__=MagicMock(return_value=dst), __exit__=MagicMock(return_value=False)),
            ]
            with patch('src.papyrus_scripts.subsim_search.Path.replace'), \
                 patch('src.papyrus_scripts.subsim_search.Path.unlink'), \
                 patch('src.papyrus_scripts.subsim_search.Path.is_file', return_value=False):
                ss.sort_db_file('fake.h5', verbose=progress)

        skipped_group._f_copy.assert_not_called()
        simfp_group._f_copy.assert_called_once()
        other_group_node._f_copy.assert_called_once()
        array_node.copy.assert_called_once()
        self.assertTrue(any('create_vlarray' in str(c) for c in dst.mock_calls))

    def test_sorts_with_progress(self):
        self._run(progress=True)

    def test_sorts_without_progress(self):
        self._run(progress=False)

    def test_removes_stale_tmp_file_before_sorting(self):
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'calc_popcnt_bins_pytables', create=True, return_value=[]),
            patch('src.papyrus_scripts.subsim_search.Path.is_file', return_value=True) as mock_is_file,
            patch('src.papyrus_scripts.subsim_search.Path.unlink') as mock_unlink,
            patch('src.papyrus_scripts.subsim_search.Path.replace'),
        ):
            src = MagicMock()
            src.walk_groups.return_value = []
            src.iter_nodes.return_value = []
            dst = MagicMock()
            mock_tb.open_file.side_effect = [
                MagicMock(__enter__=MagicMock(return_value=src), __exit__=MagicMock(return_value=False)),
                MagicMock(__enter__=MagicMock(return_value=dst), __exit__=MagicMock(return_value=False)),
            ]
            ss.sort_db_file('fake.h5')
        self.assertTrue(mock_is_file.called)
        self.assertTrue(mock_unlink.called)

    def test_failure_removes_temp_and_leaves_original_untouched(self):
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch('src.papyrus_scripts.subsim_search.Path.is_file', return_value=False),
            patch('src.papyrus_scripts.subsim_search.Path.unlink') as mock_unlink,
            patch('src.papyrus_scripts.subsim_search.Path.replace') as mock_replace,
        ):
            mock_tb.open_file.side_effect = RuntimeError('boom')
            with self.assertRaises(RuntimeError):
                ss.sort_db_file('fake.h5')
        mock_unlink.assert_called_once_with(missing_ok=True)
        mock_replace.assert_not_called()


class TestFPSubSim2Init(unittest.TestCase):

    def test_raises_when_deps_missing(self):
        with patch.object(ss, 'HAS_TABLES', False), patch.object(ss, 'HAS_FPSIM2', False):
            with self.assertRaises(ImportError):
                FPSubSim2()

    def test_sets_defaults_when_deps_available(self):
        with patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True):
            engine = FPSubSim2()
        self.assertIsNone(engine.version)
        self.assertIsNone(engine.is3d)
        self.assertIsNone(engine.sd_file)
        self.assertIsNone(engine.h5_filename)


class TestCreateFromPapyrus(unittest.TestCase):
    """PapyrusVersion is used in an isinstance() check, so it can't be
    replaced wholesale by a MagicMock - only its is_downloaded() result is
    stubbed, on a real (offline-constructible) instance.
    """

    def test_raises_when_version_not_downloaded(self):
        engine = make_engine()
        with (
            patch('src.papyrus_scripts.utils.IO.PapyrusVersion.is_downloaded', return_value=False),
        ):
            with self.assertRaises(ValueError):
                engine.create_from_papyrus()

    def test_delegates_to_create_with_located_sd_file(self):
        engine = make_engine()
        with (
            patch('src.papyrus_scripts.utils.IO.PapyrusVersion.is_downloaded', return_value=True),
            patch('src.papyrus_scripts.subsim_search._set_root_folder') as mock_set_root,
            patch('src.papyrus_scripts.subsim_search.pystow.join', return_value='/fake/structures'),
            patch('src.papyrus_scripts.subsim_search.locate_file', return_value=[Path('/fake/mols.sd.xz')]),
            patch('src.papyrus_scripts.subsim_search.get_num_rows_in_file', return_value=42),
            patch.object(engine, 'create') as mock_create,
        ):
            engine.create_from_papyrus(is3d=True, outfile='out.h5', njobs=2, progress=False)

        mock_set_root.assert_called_once_with(None)
        self.assertEqual(engine.sd_file, Path('/fake/mols.sd.xz'))
        mock_create.assert_called_once_with(
            sd_file=Path('/fake/mols.sd.xz'), outfile='out.h5', fingerprint=ANY,
            total=42, progress=False, njobs=2,
            n_shards=None, pattern_holder_bits=ss.DEFAULT_PATTERN_HOLDER_BITS,
        )


class TestCreate(unittest.TestCase):

    def test_invalid_njobs_raises(self):
        engine = make_engine()
        with self.assertRaises(ValueError):
            engine.create(sd_file='mols.sd', fingerprint=[MorganFingerprint()], njobs=-2)

    def test_single_process_dispatch(self):
        engine = make_engine()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'create_schema', create=True),
            patch.object(engine, '_single_process_create') as mock_single,
            patch.object(engine, '_parallel_create') as mock_parallel,
            patch.object(engine, 'load') as mock_load,
            patch.object(Path, 'replace'),
        ):
            mock_tb.Filters.return_value = MagicMock()
            mock_tb.open_file.return_value.__enter__.return_value = MagicMock()
            engine.create(sd_file='mols.sd', fingerprint=[MorganFingerprint()], njobs=1, progress=False)
        mock_single.assert_called_once()
        mock_parallel.assert_not_called()
        mock_load.assert_called_once_with(engine.h5_filename)

    def test_parallel_dispatch(self):
        engine = make_engine()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'create_schema', create=True),
            patch.object(engine, '_single_process_create') as mock_single,
            patch.object(engine, '_parallel_create') as mock_parallel,
            patch.object(engine, 'load'),
            patch.object(Path, 'replace'),
        ):
            mock_tb.Filters.return_value = MagicMock()
            mock_tb.open_file.return_value.__enter__.return_value = MagicMock()
            engine.create(sd_file='mols.sd', fingerprint=[MorganFingerprint()], njobs=3, progress=False)
        mock_parallel.assert_called_once()
        mock_single.assert_not_called()

    def test_default_outfile_name_uses_version_and_dimension(self):
        engine = make_engine()
        engine.is3d = True
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'create_schema', create=True),
            patch.object(engine, '_single_process_create'),
            patch.object(engine, 'load'),
            patch.object(Path, 'replace'),
        ):
            mock_tb.Filters.return_value = MagicMock()
            mock_tb.open_file.return_value.__enter__.return_value = MagicMock()
            engine.create(sd_file='mols.sd', fingerprint=[MorganFingerprint()], njobs=1, progress=False)
        self.assertEqual(engine.h5_filename, Path('Papyrus_custom_FPSubSim2_3D.h5'))

    def test_build_failure_cleans_up_temp_file_and_final_path_untouched(self):
        engine = make_engine()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'create_schema', create=True),
            patch.object(engine, '_single_process_create', side_effect=RuntimeError('boom')),
            patch.object(Path, 'glob') as mock_glob,
        ):
            mock_tb.Filters.return_value = MagicMock()
            mock_tb.open_file.return_value.__enter__.return_value = MagicMock()
            leftover = MagicMock()
            mock_glob.return_value = [leftover]
            with self.assertRaises(RuntimeError):
                engine.create(sd_file='mols.sd', fingerprint=[MorganFingerprint()], njobs=1, progress=False)
        leftover.unlink.assert_called_once_with(missing_ok=True)
        self.assertEqual(engine.h5_filename, Path('Papyrus_custom_FPSubSim2_2D.h5.building'))


class TestLoad(unittest.TestCase):

    def test_raises_when_file_missing(self):
        engine = make_engine()
        with self.assertRaises(ValueError):
            engine.load('/does/not/exist.h5')

    def test_reconstructs_version_and_dim(self):
        engine = make_engine()
        with (
            patch('src.papyrus_scripts.subsim_search.Path.is_file', return_value=True),
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch('src.papyrus_scripts.subsim_search.PapyrusVersion') as mock_version_cls,
        ):
            fake_h5 = MagicMock()
            fake_h5.root.config.read.return_value = [(ss.rdkit.__version__, '2024.09.2', '2D')]
            mock_tb.open_file.return_value.__enter__.return_value = fake_h5
            engine.load('fake.h5')
        self.assertFalse(engine.is3d)
        mock_version_cls.assert_called_once_with(version='2024.09.2')
        self.assertIsNone(engine._avail_fp)

    def test_empty_version_string_leaves_version_none(self):
        engine = make_engine()
        with (
            patch('src.papyrus_scripts.subsim_search.Path.is_file', return_value=True),
            patch.object(ss, 'tb', create=True) as mock_tb,
        ):
            fake_h5 = MagicMock()
            fake_h5.root.config.read.return_value = [(ss.rdkit.__version__, '', '3D')]
            mock_tb.open_file.return_value.__enter__.return_value = fake_h5
            engine.load('fake.h5')
        self.assertIsNone(engine.version)
        self.assertTrue(engine.is3d)

    def test_rdkit_version_mismatch_warns(self):
        engine = make_engine()
        with (
            patch('src.papyrus_scripts.subsim_search.Path.is_file', return_value=True),
            patch.object(ss, 'tb', create=True) as mock_tb,
        ):
            fake_h5 = MagicMock()
            fake_h5.root.config.read.return_value = [('0.0.0-not-real', '', '2D')]
            mock_tb.open_file.return_value.__enter__.return_value = fake_h5
            with self.assertWarns(UserWarning):
                engine.load('fake.h5')


class TestAvailableFingerprints(unittest.TestCase):

    def test_reads_and_caches(self):
        engine = make_engine()
        engine._avail_fp = None
        group = _FakeGroup('Morgan_2048bits_x')
        skipped = _FakeGroup('')
        fp_table = MagicMock()
        fp_table.attrs.fp_id = 'Morgan_2048bits_x'
        fp_table.attrs.fp_type = 'Morgan'
        fp_table.attrs.fp_params = '{"radius": 2, "nBits": 2048, "invariants": [], ' \
                                    '"fromAtoms": [], "useChirality": false, ' \
                                    '"useBondTypes": true, "useFeatures": false}'
        h5file = MagicMock()
        h5file.walk_groups.return_value = [skipped, group]
        h5file.get_node.return_value = fp_table
        with patch.object(ss, 'tb', create=True) as mock_tb:
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            result1 = engine.available_fingerprints
            result2 = engine.available_fingerprints
        self.assertIn('Morgan_2048bits_x', result1)
        self.assertIs(result1, result2)
        h5file.get_node.assert_called_once()  # cached on 2nd access


class TestGetSubstructureLib(unittest.TestCase):

    def test_raises_when_no_h5_file(self):
        engine = make_engine()
        engine.h5_filename = None
        with self.assertRaises(ValueError):
            engine.get_substructure_lib()

    def test_raises_when_h5_file_does_not_exist(self):
        engine = make_engine()
        engine.h5_filename = Path('/does/not/exist.h5')
        with self.assertRaises(ValueError):
            engine.get_substructure_lib()

    def test_deserializes_legacy_single_blob_library(self):
        from rdkit.Chem.rdSubstructLibrary import CachedMolHolder, PatternHolder, SubstructLibrary
        lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
        lib.AddMol(Chem.MolFromSmiles('CCO'))
        raw = _pad_to_int64(_serialize_substruct_lib(lib))
        padding = _padding_for(_serialize_substruct_lib(lib))
        ints = np.frombuffer(raw, dtype=np.int64)

        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        h5file = MagicMock()
        # Must set explicitly: MagicMock auto-creates attrs, so getattr(..., default) never falls back.
        h5file.root.substructure_info._v_attrs.format_version = 1
        h5file.root.substructure_info.substruct_lib.attrs.padding = padding
        h5file.root.substructure_info.substruct_lib.read.return_value = ints
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(Path, 'is_file', return_value=True),
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            result = engine.get_substructure_lib()
        self.assertEqual(result.GetMatches.__self__.__class__.__name__, 'PapyrusSubstructureLibrary')
        self.assertEqual(result.fp_filename, Path('fake.h5'))

    def test_deserializes_sharded_library(self):
        from rdkit.Chem.rdSubstructLibrary import CachedSmilesMolHolder, PatternHolder, SubstructLibrary
        shard_libs = []
        for smi in ('CCO', 'c1ccccc1'):
            lib = SubstructLibrary(CachedSmilesMolHolder(), PatternHolder())
            lib.AddMol(Chem.MolFromSmiles(smi))
            shard_libs.append(lib)

        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        h5file = MagicMock()
        h5file.root.substructure_info._v_attrs.format_version = 2
        h5file.root.substructure_info._v_attrs.n_shards = 2

        def fake_get_node(path):
            shard_idx = int(path.split('shard_')[1].split('/')[0])
            node = MagicMock()
            if path.endswith('mol_ids'):
                node.read.return_value = np.array([shard_idx + 1], dtype=np.int64)
                return node
            lib_bytes = _serialize_substruct_lib(shard_libs[shard_idx])
            padded = _pad_to_int64(lib_bytes)
            node.attrs.padding = _padding_for(lib_bytes)
            node.read.return_value = np.frombuffer(padded, dtype=np.int64)
            return node

        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(Path, 'is_file', return_value=True),
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            h5file.get_node.side_effect = fake_get_node
            result = engine.get_substructure_lib()
        self.assertIsInstance(result, ss.PapyrusShardedSubstructureLibrary)
        self.assertEqual(result.n_shards, 2)
        self.assertEqual(result.fp_filename, Path('fake.h5'))


class TestGetSimilarityLib(unittest.TestCase):

    def test_raises_when_no_h5_file(self):
        engine = make_engine()
        engine.h5_filename = None
        with self.assertRaises(ValueError):
            engine.get_similarity_lib()

    def test_raises_when_signature_unknown(self):
        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        with (
            patch.object(Path, 'is_file', return_value=True),
            patch.object(FPSubSim2, 'available_fingerprints', new_callable=lambda: property(lambda self: {'a': 1})),
        ):
            with self.assertRaises(ValueError):
                engine.get_similarity_lib(fp_signature='unknown')

    def test_defaults_to_first_available_and_builds_cpu_engine(self):
        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        with (
            patch.object(Path, 'is_file', return_value=True),
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
            ),
            patch.object(ss, 'FPSubSim2Engine', create=True) as mock_engine_cls,
        ):
            result = engine.get_similarity_lib()
        mock_engine_cls.assert_called_once_with(
            Path('fake.h5'), 'sig1', in_memory_fps=True, fps_sort=False,
        )
        self.assertIs(result, mock_engine_cls.return_value)

    def test_passes_through_in_memory_fps_and_fps_sort(self):
        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        with (
            patch.object(Path, 'is_file', return_value=True),
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
            ),
            patch.object(ss, 'FPSubSim2Engine', create=True) as mock_engine_cls,
        ):
            engine.get_similarity_lib(in_memory_fps=False, fps_sort=True)
        mock_engine_cls.assert_called_once_with(
            Path('fake.h5'), 'sig1', in_memory_fps=False, fps_sort=True,
        )

    def test_cuda_flag_selects_cuda_engine(self):
        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        with (
            patch.object(Path, 'is_file', return_value=True),
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
            ),
            patch.object(ss, 'FPSubSim2CudaEngine', create=True) as mock_engine_cls,
        ):
            engine.get_similarity_lib(cuda=True)
        mock_engine_cls.assert_called_once_with(Path('fake.h5'), 'sig1')

    def test_cuda_auto_uses_cuda_engine_when_available(self):
        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        with (
            patch.object(Path, 'is_file', return_value=True),
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
            ),
            patch.object(ss, 'FPSubSim2CudaEngine', create=True) as mock_cuda_cls,
            patch.object(ss, 'FPSubSim2Engine', create=True) as mock_cpu_cls,
        ):
            result = engine.get_similarity_lib(cuda='auto')
        mock_cuda_cls.assert_called_once_with(Path('fake.h5'), 'sig1')
        mock_cpu_cls.assert_not_called()
        self.assertIs(result, mock_cuda_cls.return_value)

    def test_cuda_auto_falls_back_to_cpu_engine_with_warning(self):
        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        with (
            patch.object(Path, 'is_file', return_value=True),
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
            ),
            patch.object(ss, 'FPSubSim2CudaEngine', create=True, side_effect=ImportError('no cupy')),
            patch.object(ss, 'FPSubSim2Engine', create=True) as mock_cpu_cls,
        ):
            with self.assertWarns(UserWarning):
                result = engine.get_similarity_lib(cuda='auto')
        mock_cpu_cls.assert_called_once_with(
            Path('fake.h5'), 'sig1', in_memory_fps=True, fps_sort=False,
        )
        self.assertIs(result, mock_cpu_cls.return_value)


class TestAddFingerprint(unittest.TestCase):

    def test_skips_when_already_available(self):
        engine = make_engine()
        with patch.object(
            FPSubSim2, 'available_fingerprints',
            new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
        ):
            engine.add_fingerprint(fingerprint=MagicMock(__repr__=lambda self: 'sig1'), papyrus_sd_file='mols.sd')

    def test_appends_new_fingerprint_and_invalidates_cache(self):
        engine = make_engine()
        engine._avail_fp = {'sig1': MagicMock()}
        fp = MorganFingerprint()
        with (
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
            ),
            patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls,
            patch.object(ss, 'MolSupplier'),
        ):
            engine.add_fingerprint(fingerprint=fp, papyrus_sd_file='mols.sd', progress=False, total=10)
        mock_backend_cls.return_value.change_fp_for_append.assert_called_once_with(fp)
        mock_backend_cls.return_value.append_fps.assert_called_once()
        self.assertIsNone(engine._avail_fp)


class TestAddMolecules(unittest.TestCase):

    def _patch_mol_supplier(self, mols):
        # Yields the same (mol_id, rdmol) pairs regardless of call-site kwargs.
        return patch.object(ss, 'MolSupplier', side_effect=lambda *a, **kw: MagicMock(
            __enter__=MagicMock(return_value=iter(mols)),
            __exit__=MagicMock(return_value=False),
            __iter__=lambda self: iter(mols),
        ))

    def test_legacy_format_appends_mappings_fingerprints_and_substruct_lib(self):
        from rdkit.Chem.rdSubstructLibrary import CachedMolHolder, PatternHolder, SubstructLibrary
        engine = make_engine()
        mol = Chem.MolFromSmiles('CCO')

        fake_lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
        fake_lib.fp_filename = 'fake.h5'
        fake_lib.AddMol = MagicMock(wraps=fake_lib.AddMol)

        h5file = MagicMock()
        h5file.root.mol_mappings.iterrows.return_value = iter([{'idnumber': 5}])
        h5file.root.substructure_info._v_attrs.format_version = 1

        with (
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
            ),
            patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls,
            self._patch_mol_supplier([(6, mol)]),
            patch.object(engine, 'get_substructure_lib', return_value=fake_lib),
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'sort_db_file') as mock_sort,
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            engine.add_molecules(papyrus_sd_file='mols.sd', progress=False, total=1)

        h5file.root.mol_mappings.append.assert_called_once_with([(6, ANY, ANY)])
        mock_backend_cls.return_value.append_fps.assert_called_once()
        fake_lib.AddMol.assert_called_once_with(mol)
        h5file.remove_node.assert_called_once()
        mock_sort.assert_called_once()

    def test_sharded_format_appends_only_touched_shards(self):
        engine = make_engine()
        mol1 = Chem.MolFromSmiles('CCO')
        mol2 = Chem.MolFromSmiles('c1ccccc1')

        fake_shard0, fake_shard1 = MagicMock(), MagicMock()
        fake_shard0.Serialize.return_value = b'\x01' * 8
        fake_shard1.Serialize.return_value = b'\x02' * 8
        fake_sharded_lib = MagicMock()
        fake_sharded_lib.n_shards = 2
        fake_sharded_lib._resident = [fake_shard0, fake_shard1]

        h5file = MagicMock()
        h5file.root.mol_mappings.iterrows.return_value = iter([])
        h5file.root.substructure_info._v_attrs.format_version = 2

        with (
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {}),
            ),
            self._patch_mol_supplier([(1, mol1), (2, mol2)]),
            patch.object(engine, 'get_substructure_lib', return_value=fake_sharded_lib),
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'sort_db_file') as mock_sort,
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            engine.add_molecules(papyrus_sd_file='mols.sd', progress=False, total=2)

        fake_shard0.AddMol.assert_called_once_with(mol2)  # mol_id 2 % 2 == 0
        fake_shard1.AddMol.assert_called_once_with(mol1)  # mol_id 1 % 2 == 1
        mock_sort.assert_called_once()


class TestReaderProcess(unittest.TestCase):

    def test_routes_to_shard_queues_and_sends_end_sentinels(self):
        mol = Chem.MolFromSmiles('CCO')
        input_queues = [MagicMock(), MagicMock(), MagicMock()]
        with patch.object(ss, 'MolSupplier') as mock_supplier_cls:
            # mol_id 1 -> shard 1, mol_id 2 -> shard 2, mol_id 4 -> shard 1 (4 % 3 == 1)
            mock_supplier_cls.return_value.__enter__.return_value = iter([(1, mol), (2, mol), (4, mol)])
            mock_supplier_cls.return_value.__exit__.return_value = False
            ss._reader_process('mols.sd', n_shards=3, total=3, progress=False, input_queues=input_queues)

        shard1_ids = [
            m[0] for c in input_queues[1].put.call_args_list if c.args[0] != 'END' for m in c.args[0]
        ]
        shard2_ids = [
            m[0] for c in input_queues[2].put.call_args_list if c.args[0] != 'END' for m in c.args[0]
        ]
        self.assertEqual(shard1_ids, [1, 4])
        self.assertEqual(shard2_ids, [2])
        for q in input_queues:
            self.assertTrue(any(c.args[0] == 'END' for c in q.put.call_args_list))


class TestWorkerProcess(unittest.TestCase):

    def test_processes_one_batch_then_stops_on_end(self):
        class _FakeFp:
            def __init__(self, **params):
                pass

            def __repr__(self):
                return 'FakeFp_1bits_0x0'

            def get(self, mol):
                return [0, 1]

        mol = Chem.MolFromSmiles('CCO')
        input_queue = MagicMock()
        input_queue.get.side_effect = [[(1, mol, mol.GetPropsAsDict())], 'END']
        fp_output_queue = MagicMock()
        shard_result_queue = MagicMock()
        fp_specs = [(_FakeFp, {})]

        ss._worker_process(0, fp_specs, 2048, input_queue, fp_output_queue, shard_result_queue)

        fp_output_queue.put.assert_called_once()
        kind, mappings_batch, fps_batch = fp_output_queue.put.call_args.args[0]
        self.assertEqual(kind, 'rows')
        self.assertEqual(mappings_batch, [(1, ANY, ANY)])
        self.assertIn('FakeFp_1bits_0x0', fps_batch)

        shard_result_queue.put.assert_called_once()
        shard_idx, lib_bytes, padding, mol_ids = shard_result_queue.put.call_args.args[0]
        self.assertEqual(shard_idx, 0)
        self.assertEqual(mol_ids, [1])
        self.assertIsInstance(lib_bytes, bytes)


class TestParquetShardWorkerProcess(unittest.TestCase):
    """Each worker reads its own shard's rows directly from the Parquet file."""

    class _FakeFp:
        def __init__(self, **params):
            pass

        def __repr__(self):
            return 'FakeFp_1bits_0x0'

        def get(self, mol):
            return [0, 1]

    def _write_parquet(self, path, n_mols):
        import pyarrow as pa
        import pyarrow.parquet as pq
        mol = Chem.MolFromSmiles('CCO')
        ctab = Chem.MolToMolBlock(mol)
        table = pa.table({
            'ctab': [ctab] * n_mols,
            'InChIKey': [f'KEY{i}' for i in range(n_mols)],
        })
        pq.write_table(table, path)

    def test_only_own_shard_rows_are_parsed_and_added(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'mols.parquet'
            self._write_parquet(path, n_mols=5)  # mol_ids 1..5

            fp_output_queue = MagicMock()
            shard_result_queue = MagicMock()
            fp_specs = [(self._FakeFp, {})]

            # n_shards=2: shard 0 -> mol_ids [2, 4] (mol_id % 2 == 0); shard 1 -> [1, 3, 5]
            ss._parquet_shard_worker_process(
                0, fp_specs, 2048, str(path), 2, fp_output_queue, shard_result_queue,
            )

        fp_output_queue.put.assert_called_once()
        kind, mappings_batch, fps_batch = fp_output_queue.put.call_args.args[0]
        self.assertEqual(kind, 'rows')
        self.assertEqual([m[0] for m in mappings_batch], [2, 4])
        self.assertIn('FakeFp_1bits_0x0', fps_batch)

        shard_result_queue.put.assert_called_once()
        shard_idx, lib_bytes, padding, mol_ids = shard_result_queue.put.call_args.args[0]
        self.assertEqual(shard_idx, 0)
        self.assertEqual(mol_ids, [2, 4])
        self.assertIsInstance(lib_bytes, bytes)

    def test_unparseable_ctab_is_skipped_and_warned(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'mols.parquet'
            table = pa.table({'ctab': [None], 'InChIKey': ['KEY0']})
            pq.write_table(table, path)

            fp_output_queue = MagicMock()
            shard_result_queue = MagicMock()

            with self.assertWarns(UserWarning):
                ss._parquet_shard_worker_process(
                    0, [], 2048, str(path), 1, fp_output_queue, shard_result_queue,
                )

        fp_output_queue.put.assert_not_called()
        shard_result_queue.put.assert_called_once()
        _, _, _, mol_ids = shard_result_queue.put.call_args.args[0]
        self.assertEqual(mol_ids, [])


class TestFpWriterProcess(unittest.TestCase):

    def test_flushes_on_stop_and_sets_done_event(self):
        fp_output_queue = MagicMock()
        fp_output_queue.get.side_effect = [
            ('rows', [(1, 'CONN', 'INCHI')], {'sig1': [(1, 0, 1)]}),
            'STOP',
        ]
        h5file = MagicMock()
        fp_writer_done = MagicMock()
        with patch.object(ss, 'tb', create=True) as mock_tb:
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            ss._fp_writer_process(
                'fake.h5', fp_output_queue, table_paths={'sig1': '/similarity_info/sig1/fps'},
                progress_queue=None, fp_writer_done=fp_writer_done,
            )
        h5file.root.mol_mappings.append.assert_called()
        h5file.get_node.return_value.append.assert_called()
        fp_writer_done.set.assert_called_once()

    def test_batch_flush_mid_stream(self):
        messages = [
            ('rows', [(i, 'CONN', 'INCHI') for i in range(3)], {'sig1': [(i, 0, 1) for i in range(3)]}),
            'STOP',
        ]
        fp_output_queue = MagicMock()
        fp_output_queue.get.side_effect = messages
        h5file = MagicMock()
        fp_writer_done = MagicMock()
        progress_queue = MagicMock()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'BATCH_WRITE_SIZE', 2, create=True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            ss._fp_writer_process(
                'fake.h5', fp_output_queue, table_paths={'sig1': '/similarity_info/sig1/fps'},
                progress_queue=progress_queue, fp_writer_done=fp_writer_done,
            )
        self.assertGreaterEqual(h5file.root.mol_mappings.append.call_count, 1)
        self.assertGreaterEqual(h5file.get_node.return_value.append.call_count, 1)
        progress_queue.put.assert_called_once_with(3)
        fp_writer_done.set.assert_called_once()

    def test_sets_done_event_even_on_exception(self):
        fp_output_queue = MagicMock()
        fp_output_queue.get.side_effect = RuntimeError('boom')
        fp_writer_done = MagicMock()
        with patch.object(ss, 'tb', create=True) as mock_tb:
            mock_tb.open_file.return_value.__enter__.return_value = MagicMock()
            with self.assertRaises(RuntimeError):
                ss._fp_writer_process(
                    'fake.h5', fp_output_queue, table_paths={},
                    progress_queue=None, fp_writer_done=fp_writer_done,
                )
        fp_writer_done.set.assert_called_once()


class TestSubstWriterProcess(unittest.TestCase):

    def test_waits_for_fp_writer_done_before_writing(self):
        shard_result_queue = MagicMock()
        shard_result_queue.get.side_effect = [
            (0, b'\x01' * 8, 0, [1, 3]),
            (1, b'\x02' * 8, 0, [2, 4]),
        ]
        fp_writer_done = MagicMock()
        h5file = MagicMock()
        with patch.object(ss, 'tb', create=True) as mock_tb:
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            ss._subst_writer_process('fake.h5', shard_result_queue, n_shards=2, fp_writer_done=fp_writer_done)

        fp_writer_done.wait.assert_called_once()
        # 2 shards x (substruct_lib append + mol_ids append)
        self.assertEqual(h5file.get_node.return_value.append.call_count, 4)


class TestSingleProcessCreate(unittest.TestCase):

    def _run(self, n_mols, batch_write_size, n_shards=2, pattern_holder_bits=2048):
        mols = [Chem.MolFromSmiles('CCO') for _ in range(n_mols)]
        engine = make_engine()
        h5file = MagicMock()

        class _FakeFp:
            length = 8

            def __repr__(self):
                return 'FakeFp'

            def get(self, mol):
                return [0, 1]

        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'BATCH_WRITE_SIZE', batch_write_size, create=True),
            patch.object(ss, 'MolSupplier') as mock_supplier_cls,
            patch.object(ss, 'sort_db_file') as mock_sort,
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            mock_supplier_cls.return_value.__enter__.return_value = enumerate(mols, 1)
            mock_supplier_cls.return_value.__exit__.return_value = False
            engine._single_process_create(
                fingerprint=[_FakeFp()], progress=False, total=n_mols,
                n_shards=n_shards, pattern_holder_bits=pattern_holder_bits,
            )

        mock_sort.assert_called_once()
        return h5file

    def test_flushes_final_partial_batch(self):
        h5file = self._run(n_mols=1, batch_write_size=100)
        h5file.root.mol_mappings.append.assert_called_once()
        h5file.get_node.return_value.append.assert_called()

    def test_flushes_full_batches_mid_stream(self):
        h5file = self._run(n_mols=4, batch_write_size=2)
        self.assertEqual(h5file.root.mol_mappings.append.call_count, 2)


class TestPyTablesMultiFpStorageBackend(unittest.TestCase):
    """BaseStorageBackend is FPSim2's own base class - a no-op stub here
    since FPSim2 isn't installed, so its __init__ must be patched to accept
    fp_filename (the stub falls back to object.__init__, which doesn't).
    """

    def setUp(self):
        patcher = patch.object(
            ss.BaseStorageBackend, '__init__',
            new=lambda self, fp_filename: setattr(self, 'fp_filename', fp_filename),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _make_h5file(self, fp_signature='sig1'):
        group = _FakeGroup(fp_signature)
        fp_table = MagicMock()
        fp_table.attrs.fp_id = fp_signature
        fp_table.attrs.fp_type = 'Morgan'
        fp_table.attrs.fp_params = '{}'
        fp_table.chunkshape = (10,)
        h5file = MagicMock()
        h5file.walk_groups.return_value = [_FakeGroup(''), group]
        h5file.get_node.return_value = fp_table
        h5file.root.config = [b'rdkit-ver']
        return h5file

    def test_raises_when_signature_unknown(self):
        h5file = self._make_h5file()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            with self.assertRaises(ValueError):
                ss.PyTablesMultiFpStorageBackend('fake.h5', 'unknown', in_memory_fps=False)

    def test_init_loads_popcounts_without_in_memory_fps(self):
        h5file = self._make_h5file()
        h5file.get_node.return_value.__getitem__.return_value = np.zeros(3)
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True) as mock_get_fp,
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=False)
        self.assertEqual(backend.name, 'pytables')
        mock_get_fp.assert_called_once()

    def test_read_parameters(self):
        h5file = self._make_h5file()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=False)
            fp_type, fp_params, rdkit_ver = backend.read_parameters()
        self.assertEqual(fp_type, 'Morgan')
        self.assertEqual(fp_params, {})

    def test_get_fps_chunk(self):
        h5file = self._make_h5file()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=False)
            backend.get_fps_chunk((0, 10))
        h5file.get_node.return_value.__getitem__.assert_called()

    def test_load_fps_reshapes_view(self):
        h5file = self._make_h5file()
        fps = np.zeros(3, dtype=[('mol_id', '<u8'), ('bit0', '<u8')])
        h5file.get_node.return_value.__getitem__.return_value = fps
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=True)
        self.assertEqual(backend.fps.shape[1], 2)

    def test_load_fps_and_popcnt_bins_sort_branch(self):
        # calc_popcnt_bins is inherited from FPSim2's real BaseStorageBackend
        # (not installed here, stubbed) - patch it directly onto the stub.
        h5file = self._make_h5file()
        fps = np.zeros(3, dtype=[('mol_id', '<u8'), ('bit0', '<u8'), ('popcnt', '<u8')])
        h5file.get_node.return_value.__getitem__.return_value = fps
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True),
            patch.object(ss.BaseStorageBackend, 'calc_popcnt_bins', create=True, return_value=[b'\x00']),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=True, fps_sort=True)
        self.assertEqual(backend.popcnt_bins, [b'\x00'])

    def test_delete_fps_removes_matching_rows(self):
        h5file = self._make_h5file()
        row = MagicMock()
        row.nrow = 5
        fps_table = h5file.get_node.return_value
        fps_table.where.return_value = iter([row])
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=False)
            backend.delete_fps([42])
        fps_table.remove_row.assert_called_once_with(5)

    def test_append_fps_flushes_full_batches_and_sorts(self):
        h5file = self._make_h5file()
        fps_table = h5file.get_node.return_value
        fps_table.iterrows.return_value = iter([{'fp_id': 5}])
        mol = Chem.MolFromSmiles('CCO')
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True) as mock_get_fp,
            patch.object(ss, 'BATCH_WRITE_SIZE', 1, create=True),
            patch.object(ss, 'sort_db_file') as mock_sort,
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            mock_get_fp.return_value.get.return_value = [0, 1]
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=False)
            supplier = MagicMock()
            supplier.set_start_progress_total = MagicMock()
            supplier.__iter__.return_value = iter([(1, mol), (2, mol)])
            backend.append_fps(supplier, progress=False, total=2, sort=True)
        # +1: continues after the last existing fp_id (5) rather than reusing it.
        supplier.set_start_progress_total.assert_called_once_with(6, False, 2)
        mock_sort.assert_called_once()

    def test_append_fps_final_partial_batch_without_sort(self):
        h5file = self._make_h5file()
        fps_table = h5file.get_node.return_value
        fps_table.iterrows.return_value = iter([])  # empty table -> default start_id=1
        mol = Chem.MolFromSmiles('CCO')
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True) as mock_get_fp,
            patch.object(ss, 'BATCH_WRITE_SIZE', 100, create=True),
            patch.object(ss, 'sort_db_file') as mock_sort,
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            mock_get_fp.return_value.get.return_value = [0, 1]
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=False)
            supplier = MagicMock()
            supplier.__iter__.return_value = iter([(1, mol)])
            backend.append_fps(supplier, progress=False, total=1, sort=False)
        fps_table.append.assert_called_once()
        mock_sort.assert_not_called()

    def test_change_fp_for_append_creates_new_table(self):
        h5file = self._make_h5file()
        new_group = MagicMock()
        new_group._v_name = 'NewFp'
        h5file.create_group.return_value = new_group
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'get_fp_from_name', create=True),
            patch.object(ss, 'create_schema', create=True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            backend = ss.PyTablesMultiFpStorageBackend('fake.h5', 'sig1', in_memory_fps=False)
            backend.change_fp_for_append(MorganFingerprint())
        self.assertEqual(backend._current_fp_path, '/similarity_info/NewFp/fps')


class TestMappingMixinGetMapping(unittest.TestCase):

    def _make_mixin(self):
        obj = _MappingMixin()
        obj.fp_filename = 'fake.h5'
        return obj

    _MAPPING_DTYPE = [('idnumber', '<i8'), ('connectivity', 'S4'), ('InChIKey', 'S5')]

    def _mock_mappings_table(self, rows: list[tuple]) -> MagicMock:
        mappings_table = MagicMock()
        mappings_table.cols._v_colnames = ['idnumber', 'connectivity', 'InChIKey']
        mappings_table.read.return_value = np.array(rows, dtype=self._MAPPING_DTYPE)
        return mappings_table

    def test_wraps_scalar_id_in_list(self):
        obj = self._make_mixin()
        mappings_table = self._mock_mappings_table([(1, b'CONN', b'INCHI')])
        h5file = MagicMock()
        h5file.root.mol_mappings = mappings_table
        with patch.object(ss, 'tb', create=True) as mock_tb:
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            df = obj._get_mapping(1)
        self.assertEqual(df['idnumber'].to_list(), [1])
        self.assertEqual(list(df.columns), ['idnumber', 'connectivity', 'InChIKey'])

    def test_empty_ids_raises(self):
        obj = self._make_mixin()
        with self.assertRaises(ValueError):
            obj._get_mapping([])

    def test_non_integer_id_raises(self):
        obj = self._make_mixin()
        with self.assertRaises(ValueError):
            obj._get_mapping([1.5])

    def test_missing_id_raises(self):
        obj = self._make_mixin()
        mappings_table = self._mock_mappings_table([])
        h5file = MagicMock()
        h5file.root.mol_mappings = mappings_table
        with patch.object(ss, 'tb', create=True) as mock_tb:
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            with self.assertRaises(ValueError):
                obj._get_mapping([999])


class _ConcreteMultiFpEngine(ss.BaseMultiFpEngine):
    """Minimal concrete subclass - BaseMultiFpEngine is abstract (real FPSim2's
    BaseEngine declares `similarity` abstract).
    """

    def similarity(self, query_string, threshold, n_workers=1):
        raise NotImplementedError


class TestBaseMultiFpEngine(unittest.TestCase):

    def test_init_builds_pytables_backend(self):
        with patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls, \
             patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True):
            engine = _ConcreteMultiFpEngine.__new__(_ConcreteMultiFpEngine)
            ss.BaseMultiFpEngine.__init__(
                engine, 'fake.h5', 'sig1', storage_backend='pytables',
                in_memory_fps=True, fps_sort=False,
            )
        mock_backend_cls.assert_called_once_with('fake.h5', 'sig1', in_memory_fps=True, fps_sort=False)
        self.assertIs(engine.storage, mock_backend_cls.return_value)

    def test_load_query_raises_on_unparseable_molecule(self):
        engine = _ConcreteMultiFpEngine.__new__(_ConcreteMultiFpEngine)
        with patch.object(ss, 'load_molecule', create=True, return_value=None):
            with self.assertRaises(ValueError):
                engine.load_query('not a molecule')

    def test_load_query_builds_fingerprint_array(self):
        engine = _ConcreteMultiFpEngine.__new__(_ConcreteMultiFpEngine)
        engine.storage = MagicMock()
        engine.storage.fp_type = 'Morgan'
        engine.storage.fp_params = {}
        mol = Chem.MolFromSmiles('CCO')
        fake_fp = MagicMock()
        fake_fp.get.return_value = [1, 2, 3]
        with (
            patch.object(ss, 'load_molecule', create=True, return_value=mol),
            patch.object(ss, 'get_fp_from_name', create=True, return_value=fake_fp),
        ):
            result = engine.load_query('CCO')
        self.assertEqual(result.tolist(), [0, 1, 2, 3])
        self.assertEqual(result.dtype, np.uint64)


class TestFPSubSim2EngineInit(unittest.TestCase):

    def test_sets_empty_result_arrays(self):
        with patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls, \
             patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True):
            engine = ss.FPSubSim2Engine('fake.h5', 'sig1')
        mock_backend_cls.assert_called_once_with(
            'fake.h5', 'sig1', in_memory_fps=True, fps_sort=False,
        )
        self.assertEqual(engine.empty_sim.shape, (0,))
        self.assertEqual(engine.empty_subs.shape, (0,))


class TestFPSubSim2EngineMethods(unittest.TestCase):

    def _make_engine(self):
        engine = ss.FPSubSim2Engine.__new__(ss.FPSubSim2Engine)
        engine.storage = MagicMock()
        engine.storage._current_fp = 'sig1'
        engine._get_mapping = MagicMock(return_value=pl.DataFrame({
            'idnumber': [1], 'connectivity': ['a'], 'InChIKey': ['b'],
        }))
        return engine

    def test_similarity_builds_df_with_tanimoto_label(self):
        engine = self._make_engine()
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.similarity.return_value = [(1, 0.9)]
            df = engine.similarity('CCO', 0.7)
        # By keyword: FPSim2's real signature has gained positional params ahead of n_workers.
        mock_cls.similarity.assert_called_once_with(engine, 'CCO', 0.7, n_workers=1)
        self.assertTrue(any('Tanimoto' in c for c in df.columns))

    def test_on_disk_similarity(self):
        engine = self._make_engine()
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.on_disk_similarity.return_value = [(1, 0.9)]
            df = engine.on_disk_similarity('CCO', 0.7)
        mock_cls.on_disk_similarity.assert_called_once_with(
            engine, 'CCO', 0.7, n_workers=1, chunk_size=0,
        )
        self.assertTrue(any('Tanimoto' in c for c in df.columns))

    def test_tversky_builds_df_with_tversky_label(self):
        engine = self._make_engine()
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.tversky.return_value = [(1, 0.9)]
            df = engine.tversky('CCO', 0.7, a=0.5, b=0.5)
        mock_cls.tversky.assert_called_once_with(engine, 'CCO', 0.7, 0.5, 0.5, n_workers=1)
        self.assertTrue(any('Tversky' in c for c in df.columns))

    def test_on_disk_tversky(self):
        engine = self._make_engine()
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.on_disk_tversky.return_value = [(1, 0.9)]
            df = engine.on_disk_tversky('CCO', 0.7, a=0.5, b=0.5)
        mock_cls.on_disk_tversky.assert_called_once_with(
            engine, 'CCO', 0.7, 0.5, 0.5, n_workers=1, chunk_size=None,
        )
        self.assertTrue(any('Tversky' in c for c in df.columns))

    def test_similarity_handles_numpy_structured_array_results(self):
        # Guards against bool() on a multi-element numpy array (raises).
        engine = self._make_engine()
        raw = np.array([(1, 0.9)], dtype=[('mol_id', '<u4'), ('coeff', '<f4')])
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.similarity.return_value = raw
            df = engine.similarity('CCO', 0.7)
        self.assertEqual(df.height, 1)

    def test_similarity_handles_empty_numpy_structured_array_results(self):
        engine = self._make_engine()
        raw = np.array([], dtype=[('mol_id', '<u4'), ('coeff', '<f4')])
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.similarity.return_value = raw
            df = engine.similarity('CCO', 0.7)
        self.assertEqual(df.height, 0)

    def test_substructure_raises_not_implemented(self):
        engine = self._make_engine()
        with self.assertRaises(NotImplementedError):
            engine.substructure('CCO')

    def test_on_disk_substructure_raises_not_implemented(self):
        engine = self._make_engine()
        with self.assertRaises(NotImplementedError):
            engine.on_disk_substructure('CCO')


class TestFPSubSim2CudaEngineImportGuard(unittest.TestCase):

    def test_raises_without_cupy(self):
        with patch.object(ss, 'HAS_CUPY', False):
            with self.assertRaises(ImportError):
                ss.FPSubSim2CudaEngine('fake.h5', 'sig1')

    def test_raises_on_invalid_kernel(self):
        with patch.object(ss, 'HAS_CUPY', True):
            with self.assertRaises(ValueError):
                ss.FPSubSim2CudaEngine('fake.h5', 'sig1', kernel='bogus')

    @unittest.skipIf(
        hasattr(ss.FPSim2CudaEngine, 'ew_kernel'),
        'only meaningful against an FPSim2 version without an element-wise kernel',
    )
    def test_element_wise_kernel_raises_when_unsupported_by_installed_fpsim2(self):
        with patch.object(ss, 'HAS_CUPY', True):
            with self.assertRaises(NotImplementedError):
                ss.FPSubSim2CudaEngine('fake.h5', 'sig1', kernel='element_wise')

    def test_element_wise_kernel_guard_is_skipped_when_ew_kernel_exists(self):
        with (
            patch.object(ss, 'HAS_CUPY', True),
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'cupy', create=True),
            patch.object(ss.FPSim2CudaEngine, 'ew_kernel', 'fake_kernel_source', create=True),
            patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls,
        ):
            mock_backend_cls.return_value = MagicMock(spec=[])
            # Guard skipped; fails downstream instead (missing `fps`).
            with self.assertRaises(AttributeError):
                ss.FPSubSim2CudaEngine('fake.h5', 'sig1', kernel='element_wise')

    # spec=[] means the mocked storage backend has no `fps` attr, so
    # constructing the engine fails deterministically past kernel validation.
    def test_builds_storage_backend_then_fails_on_the_cupy_specific_branch(self):
        with (
            patch.object(ss, 'HAS_CUPY', True),
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'cupy', create=True),
            patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls,
        ):
            mock_backend_cls.return_value = MagicMock(spec=[])
            with self.assertRaises(AttributeError):
                ss.FPSubSim2CudaEngine('fake.h5', 'sig1', kernel='raw')
        mock_backend_cls.assert_called_once_with(
            'fake.h5', 'sig1', in_memory_fps=True, fps_sort=False,
        )


class TestPapyrusSubstructureLibraryGetMatches(unittest.TestCase):

    def _make_lib(self):
        from rdkit.Chem.rdSubstructLibrary import CachedMolHolder, PatternHolder, SubstructLibrary
        with patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True):
            lib = ss.PapyrusSubstructureLibrary.__new__(ss.PapyrusSubstructureLibrary)
        SubstructLibrary.__init__(lib, CachedMolHolder(), PatternHolder())
        lib.fp_filename = 'fake.h5'
        lib.mol_ids = None
        return lib

    def test_no_matches_returns_empty_schema(self):
        lib = self._make_lib()
        lib.AddMol(Chem.MolFromSmiles('CCO'))
        with patch.object(ss, 'load_molecule', create=True, return_value=Chem.MolFromSmiles('c1ccccc1')):
            result = lib.GetMatches('c1ccccc1')  # benzene: no substructure match in an ethanol-only lib
        self.assertEqual(result.height, 0)
        self.assertEqual(set(result.columns), {'idnumber', 'connectivity', 'InChIKey'})

    def test_matches_delegate_to_get_mapping(self):
        lib = self._make_lib()
        lib.AddMol(Chem.MolFromSmiles('CCO'))
        with (
            patch.object(ss, 'load_molecule', create=True, return_value=Chem.MolFromSmiles('CCO')),
            patch.object(ss._MappingMixin, '_get_mapping', return_value='mapped') as mock_get_mapping,
        ):
            result = lib.GetMatches('CCO')
        mock_get_mapping.assert_called_once_with([0])
        self.assertEqual(result, 'mapped')

    def test_substructure_is_alias_for_get_matches(self):
        lib = self._make_lib()
        lib.AddMol(Chem.MolFromSmiles('CCO'))
        with (
            patch.object(ss, 'load_molecule', create=True, return_value=Chem.MolFromSmiles('CCO')),
            patch.object(ss._MappingMixin, '_get_mapping', return_value='mapped') as mock_get_mapping,
        ):
            result = lib.substructure('CCO')
        mock_get_mapping.assert_called_once_with([0])
        self.assertEqual(result, 'mapped')


class TestParallelCreateWorkerCount(unittest.TestCase):
    """Regression test: n_workers = multiprocessing.cpu_count() - 2 (njobs=-1
    branch) had no floor, unlike the njobs>=0 branch's max(njobs - 1, 1).
    On a <=2-core machine this produced 0 (or negative) workers. This floor
    is now resolved in create() (which computes n_shards == n_workers before
    dispatching to _parallel_create - see _parallel_create's own docstring),
    not inside _parallel_create itself, so the regression is tested there.
    """

    def _resolved_n_shards(self, njobs, cpu_count):
        engine = make_engine()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'create_schema', create=True),
            patch.object(ss.multiprocessing, 'cpu_count', return_value=cpu_count),
            patch.object(engine, '_parallel_create') as mock_parallel,
            patch.object(engine, 'load'),
            patch.object(Path, 'replace'),
        ):
            mock_tb.Filters.return_value = MagicMock()
            mock_tb.open_file.return_value.__enter__.return_value = MagicMock()
            engine.create(sd_file='mols.sd', fingerprint=[MorganFingerprint()], njobs=njobs, progress=False)
        # _parallel_create(njobs, fingerprint, progress, total, n_shards, pattern_holder_bits)
        return mock_parallel.call_args.args[4]

    def test_njobs_minus_one_never_spawns_zero_workers_on_single_core(self):
        self.assertGreaterEqual(self._resolved_n_shards(njobs=-1, cpu_count=1), 1)

    def test_njobs_minus_one_never_spawns_zero_workers_on_dual_core(self):
        self.assertGreaterEqual(self._resolved_n_shards(njobs=-1, cpu_count=2), 1)


class TestParallelCreateCleanup(unittest.TestCase):
    """Every process still alive when _parallel_create exits - even via an
    exception (e.g. KeyboardInterrupt) during the wait - must be
    terminated, not left orphaned. Topology: reader + n_workers workers +
    fp_writer + subst_writer.
    """

    def _fake_process_factory(self, process_instances, alive, raise_on_first_join=None):
        process_count = [0]

        def fake_process(target=None, args=None, **kwargs):
            proc = MagicMock()
            proc.start = MagicMock()
            proc.is_alive = MagicMock(return_value=alive)
            proc.terminate = MagicMock()
            process_count[0] += 1
            if raise_on_first_join is not None and process_count[0] == 1:
                # Only the *first* join() call raises (the wait loop's) -
                # the finally cleanup's later join() must succeed, or its
                # loop over the remaining processes would be cut short too.
                join_call_count = [0]

                def join_side_effect(*args, **kwargs):
                    join_call_count[0] += 1
                    if join_call_count[0] == 1:
                        raise raise_on_first_join

                proc.join = MagicMock(side_effect=join_side_effect)
            else:
                proc.join = MagicMock()
            process_instances.append(proc)
            return proc

        return fake_process

    def test_clean_exit_does_not_terminate_anything(self):
        engine = make_engine()
        process_instances: list = []
        with (
            patch('src.papyrus_scripts.subsim_search.multiprocessing.cpu_count', return_value=4),
            patch(
                'src.papyrus_scripts.subsim_search.multiprocessing.Process',
                side_effect=self._fake_process_factory(process_instances, alive=False),
            ),
            # Fresh MagicMock per Queue() call, so per-queue cleanup is assertable.
            patch('src.papyrus_scripts.subsim_search.multiprocessing.Queue', side_effect=lambda *a, **kw: MagicMock()),
            patch('src.papyrus_scripts.subsim_search.sort_db_file') as mock_sort,
        ):
            engine._parallel_create(
                njobs=2, fingerprint=[], progress=False, total=None,
                n_shards=1, pattern_holder_bits=2048,
            )
        # reader + 1 worker + fp_writer + subst_writer
        self.assertEqual(len(process_instances), 4)
        for proc in process_instances:
            proc.terminate.assert_not_called()
        mock_sort.assert_called_once()

    def test_exception_during_wait_terminates_remaining_processes(self):
        engine = make_engine()
        process_instances: list = []
        with (
            patch('src.papyrus_scripts.subsim_search.multiprocessing.cpu_count', return_value=4),
            patch(
                'src.papyrus_scripts.subsim_search.multiprocessing.Process',
                # reader (the first process created) raises mid-wait,
                # simulating e.g. a KeyboardInterrupt; every process stays
                # "alive" throughout, as if nothing had a chance to clean up.
                side_effect=self._fake_process_factory(
                    process_instances, alive=True, raise_on_first_join=RuntimeError('interrupted'),
                ),
            ),
            patch('src.papyrus_scripts.subsim_search.multiprocessing.Queue', side_effect=lambda *a, **kw: MagicMock()),
            patch('src.papyrus_scripts.subsim_search.sort_db_file') as mock_sort,
        ):
            with self.assertRaises(RuntimeError):
                engine._parallel_create(
                    njobs=2, fingerprint=[], progress=False, total=None,
                    n_shards=1, pattern_holder_bits=2048,
                )
        self.assertTrue(process_instances)
        for proc in process_instances:
            proc.terminate.assert_called_once()
        mock_sort.assert_not_called()


class TestParallelCreateParquetTopology(unittest.TestCase):
    """A .parquet sd_file skips the reader process; workers read it directly."""

    def test_no_reader_process_and_workers_read_parquet_directly(self):
        engine = make_engine()
        engine.sd_file = 'fake_sd_file.parquet'
        process_instances: list = []
        targets: list = []

        def fake_process(target=None, args=None, **kwargs):
            proc = MagicMock()
            proc.is_alive = MagicMock(return_value=False)
            proc.join = MagicMock()
            targets.append(target)
            process_instances.append(proc)
            return proc

        with (
            patch('src.papyrus_scripts.subsim_search.multiprocessing.cpu_count', return_value=4),
            patch(
                'src.papyrus_scripts.subsim_search.multiprocessing.Process',
                side_effect=fake_process,
            ),
            patch('src.papyrus_scripts.subsim_search.multiprocessing.Queue', side_effect=lambda *a, **kw: MagicMock()),
            patch('src.papyrus_scripts.subsim_search.sort_db_file') as mock_sort,
        ):
            engine._parallel_create(
                njobs=2, fingerprint=[], progress=False, total=None,
                n_shards=2, pattern_holder_bits=2048,
            )

        # 2 workers + fp_writer + subst_writer - no reader.
        self.assertEqual(len(process_instances), 4)
        self.assertNotIn(ss._reader_process, targets)
        self.assertEqual(targets.count(ss._parquet_shard_worker_process), 2)
        mock_sort.assert_called_once()


class TestParallelCreateProgressReporting(unittest.TestCase):
    """pbar lives in the main process, driven by counts from progress_queue
    (not owned by the forked fp_writer child - not fork-safe there)."""

    def test_pbar_receives_counts_reported_by_fp_writer_and_terminates(self):
        engine = make_engine()
        process_instances: list = []

        # Same sequence feeds both .get(timeout=) and .get_nowait().
        items = iter([5, 3, 2])

        def _get(*_args, **_kwargs):
            try:
                return next(items)
            except StopIteration:
                raise queue.Empty from None
        progress_queue_mock = MagicMock()
        progress_queue_mock.get.side_effect = _get
        progress_queue_mock.get_nowait.side_effect = _get

        def fake_process_factory(target=None, args=None, **kwargs):
            proc = MagicMock()
            if target is ss._subst_writer_process:
                # Only .join()ed directly in real code - already finished
                # by the time `finally` checks it here.
                proc.is_alive = MagicMock(return_value=False)
            else:
                # True once (still running), then False - finally checks
                # again, so a fixed-length side_effect list would StopIteration.
                call_count = itertools.count()
                proc.is_alive = MagicMock(side_effect=lambda: next(call_count) == 0)
            process_instances.append(proc)
            return proc

        # Queue() order: input_queues (1 here), fp_output_queue,
        # shard_result_queue, progress_queue.
        queue_calls: list = []

        def fake_queue_factory(*args, **kwargs):
            queue_calls.append(None)
            return progress_queue_mock if len(queue_calls) == 4 else MagicMock()

        pbar = MagicMock()
        with (
            patch('src.papyrus_scripts.subsim_search.multiprocessing.cpu_count', return_value=4),
            patch(
                'src.papyrus_scripts.subsim_search.multiprocessing.Process',
                side_effect=fake_process_factory,
            ),
            patch(
                'src.papyrus_scripts.subsim_search.multiprocessing.Queue',
                side_effect=fake_queue_factory,
            ),
            patch('src.papyrus_scripts.subsim_search.tqdm', return_value=pbar) as mock_tqdm,
            patch('src.papyrus_scripts.subsim_search.sort_db_file') as mock_sort,
        ):
            engine._parallel_create(
                njobs=2, fingerprint=[], progress=True, total=10,
                n_shards=1, pattern_holder_bits=2048,
            )

        mock_tqdm.assert_called_once_with(total=10, smoothing=0.0, desc=ANY)
        pbar.update.assert_has_calls([call(5), call(3), call(2)])
        pbar.close.assert_called_once()
        for proc in process_instances:
            proc.terminate.assert_not_called()
        mock_sort.assert_called_once()


@unittest.skipUnless(HAS_TABLES and HAS_FPSIM2, 'requires tables and FPSim2')
class TestFPSubSim2AllFingerprints(unittest.TestCase):
    """End-to-end (unmocked): build a real .h5 db with every fingerprint type and query it."""

    SMILES = ['CCO', 'c1ccccc1', 'c1ccccc1O', 'CC(=O)O', 'CC(=O)Oc1ccccc1C(=O)O']  # last is aspirin

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.sd_file = Path(cls.tmpdir) / 'molecules.sd'
        with Chem.SDWriter(str(cls.sd_file)) as writer:
            for smi in cls.SMILES:
                writer.write(Chem.MolFromSmiles(smi))

        cls.fingerprints = [
            fp.MACCSKeysFingerprint(),
            fp.AvalonFingerprint(nBits=128),
            fp.MorganFingerprint(nBits=128),
            fp.TopologicalTorsionFingerprint(nBits=128),
            fp.AtomPairFingerprint(nBits=128),
            fp.RDKitTopologicalFingerprint(fpSize=128),
            fp.RDKPatternFingerprint(fpSize=128),
        ]
        if fp.HAS_PYBEL:
            cls.fingerprints += [fp.FP2Fingerprint(), fp.FP3Fingerprint(), fp.FP4Fingerprint()]

        cls.h5_file = Path(cls.tmpdir) / 'test.h5'
        cls.engine_db = FPSubSim2()
        cls.engine_db.create(
            sd_file=cls.sd_file, outfile=cls.h5_file,
            fingerprint=cls.fingerprints, progress=False, njobs=1,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_every_fingerprint_is_stored(self):
        available = self.engine_db.available_fingerprints
        self.assertEqual(set(available.keys()), {repr(f) for f in self.fingerprints})

    def test_every_fingerprint_finds_an_exact_self_match(self):
        # Aspirin, not CCO: CCO is too small for a non-empty TopologicalTorsion
        # fingerprint, and Tanimoto self-similarity of an all-zero vector is undefined.
        query = self.SMILES[-1]
        for f in self.fingerprints:
            with self.subTest(fingerprint=f.name):
                engine = self.engine_db.get_similarity_lib(fp_signature=repr(f))
                result = engine.similarity(query, threshold=0.999)
                self.assertGreaterEqual(result.height, 1)
                score_col = [c for c in result.columns if c.startswith('Tanimoto')][0]
                self.assertAlmostEqual(result[score_col].max(), 1.0, places=3)


if __name__ == '__main__':
    unittest.main()
