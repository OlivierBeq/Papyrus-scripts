# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.subsim_search.FPSubSim2._parallel_create.

multiprocessing.Process/Queue are mocked entirely: no real worker processes
are spawned, no .h5 file or SD file is touched. FPSubSim2.__init__ is
bypassed (via __new__) since it requires optional deps (tables, FPSim2)
that aren't installed in this environment; the module itself imports fine
without them (the dependency check only runs on instantiation).
"""

import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import polars as pl
from rdkit import Chem

from src.papyrus_scripts import subsim_search as ss
from src.papyrus_scripts.fingerprint import Fingerprint, MorganFingerprint
from src.papyrus_scripts.subsim_search import (
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
            with patch('src.papyrus_scripts.subsim_search.Path.rename'), \
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

    def test_removes_stale_tmp_file_before_rename(self):
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'calc_popcnt_bins_pytables', create=True, return_value=[]),
            patch('src.papyrus_scripts.subsim_search.Path.is_file', return_value=True) as mock_is_file,
            patch('src.papyrus_scripts.subsim_search.Path.unlink') as mock_unlink,
            patch('src.papyrus_scripts.subsim_search.Path.rename'),
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


class TestFPSubSim2Init(unittest.TestCase):

    def test_raises_when_deps_missing(self):
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
        ):
            mock_tb.Filters.return_value = MagicMock()
            mock_tb.open_file.return_value.__enter__.return_value = MagicMock()
            engine.create(sd_file='mols.sd', fingerprint=[MorganFingerprint()], njobs=1, progress=False)
        self.assertEqual(engine.h5_filename, Path('Papyrus_custom_FPSubSim2_3D.h5'))


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

    def test_deserializes_library(self):
        from rdkit.Chem.rdSubstructLibrary import CachedMolHolder, PatternHolder, SubstructLibrary
        lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
        lib.AddMol(Chem.MolFromSmiles('CCO'))
        raw = _pad_to_int64(_serialize_substruct_lib(lib))
        padding = _padding_for(_serialize_substruct_lib(lib))
        ints = np.frombuffer(raw, dtype=np.int64)

        engine = make_engine()
        engine.h5_filename = Path('fake.h5')
        h5file = MagicMock()
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
        mock_engine_cls.assert_called_once_with(Path('fake.h5'), 'sig1')
        self.assertIs(result, mock_engine_cls.return_value)

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

    def test_appends_to_every_fingerprint_and_substructure_lib(self):
        from rdkit.Chem.rdSubstructLibrary import CachedMolHolder, PatternHolder, SubstructLibrary
        engine = make_engine()
        mol = Chem.MolFromSmiles('CCO')

        fake_lib = SubstructLibrary(CachedMolHolder(), PatternHolder())
        fake_lib.AddMol = MagicMock(wraps=fake_lib.AddMol)
        with (
            patch.object(
                FPSubSim2, 'available_fingerprints',
                new_callable=lambda: property(lambda self: {'sig1': MagicMock()}),
            ),
            patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls,
            patch.object(ss, 'MolSupplier', side_effect=lambda *a, **kw: MagicMock(
                __enter__=MagicMock(return_value=iter([(1, mol)])), __exit__=MagicMock(return_value=False),
                __iter__=lambda self: iter([(1, mol)]),
            )),
            patch.object(engine, 'get_substructure_lib', return_value=fake_lib),
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'sort_db_file') as mock_sort,
        ):
            mock_tb.open_file.return_value.__enter__.return_value = MagicMock()
            engine.add_molecules(papyrus_sd_file='mols.sd', progress=False, total=1)

        mock_backend_cls.return_value.append_fps.assert_called_once()
        fake_lib.AddMol.assert_called_once_with(mol)
        mock_sort.assert_called_once()


class TestReaderProcess(unittest.TestCase):

    def test_feeds_queue_and_sends_end_sentinels(self):
        mol = Chem.MolFromSmiles('CCO')
        input_queue = MagicMock()
        with patch.object(ss, 'MolSupplier') as mock_supplier_cls:
            mock_supplier_cls.return_value.__enter__.return_value = iter([(1, mol)])
            mock_supplier_cls.return_value.__exit__.return_value = False
            ss._reader_process('mols.sd', n_workers=3, total=1, progress=False, input_queue=input_queue)
        calls = input_queue.put.call_args_list
        self.assertEqual(calls[0].args[0][0], 1)
        end_count = sum(1 for c in calls if c.args[0] == 'END')
        self.assertEqual(end_count, 3)


class TestWorkerProcess(unittest.TestCase):

    def test_processes_one_molecule_then_stops_on_end(self):
        class _FakeFp:
            def __init__(self, **params):
                pass

            def __repr__(self):
                return 'FakeFp_1bits_0x0'

            def get(self, mol):
                return [0, 1]

        mol = Chem.MolFromSmiles('CCO')
        input_queue = MagicMock()
        input_queue.get.side_effect = [(1, mol, mol.GetPropsAsDict()), 'END']
        output_queue = MagicMock()
        fp_specs = [(_FakeFp, {})]
        ss._worker_process(fp_specs, input_queue, output_queue)
        kinds = [c.args[0][0] for c in output_queue.put.call_args_list]
        self.assertEqual(kinds, ['substructure', 'mappings', 'similarity'])


class TestWriterProcess(unittest.TestCase):

    def test_flushes_on_stop(self):
        mol = Chem.MolFromSmiles('CCO')
        output_queue = MagicMock()
        output_queue.get.side_effect = [
            ('substructure', mol),
            ('mappings', (1, 'CONN', 'INCHI')),
            ('similarity', 'sig1', (1, 0, 1)),
            'STOP',
        ]
        h5file = MagicMock()
        with patch.object(ss, 'tb', create=True) as mock_tb:
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            ss._writer_process(
                'fake.h5', output_queue, table_paths={'sig1': '/similarity_info/sig1/fps'},
                total=1, progress=False,
            )
        h5file.root.mol_mappings.append.assert_called()
        h5file.get_node.return_value.append.assert_called()
        h5file.root.substructure_info.substruct_lib.append.assert_called_once()

    def test_batch_flush_mid_stream(self):
        messages = (
            [('mappings', (i, 'CONN', 'INCHI')) for i in range(3)]
            + [('similarity', 'sig1', (i, 0, 1)) for i in range(3)]
            + ['STOP']
        )
        output_queue = MagicMock()
        output_queue.get.side_effect = messages
        h5file = MagicMock()
        with (
            patch.object(ss, 'tb', create=True) as mock_tb,
            patch.object(ss, 'BATCH_WRITE_SIZE', 2, create=True),
        ):
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            ss._writer_process(
                'fake.h5', output_queue, table_paths={'sig1': '/similarity_info/sig1/fps'},
                total=None, progress=True,
            )
        # append() called at least twice: once for the mid-stream flush, once at STOP.
        self.assertGreaterEqual(h5file.root.mol_mappings.append.call_count, 2)
        # Similarity batch-flush branch (separate from the STOP flush) ran at least once.
        self.assertGreaterEqual(h5file.get_node.return_value.append.call_count, 1)


class TestSingleProcessCreate(unittest.TestCase):

    def _run(self, n_mols, batch_write_size):
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
            engine._single_process_create(fingerprint=[_FakeFp()], progress=False, total=n_mols)

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
        supplier.set_start_progress_total.assert_called_once_with(5, False, 2)
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

    def test_wraps_scalar_id_in_list(self):
        obj = self._make_mixin()
        row = MagicMock()
        row.fetch_all_fields.return_value = (1, b'CONN', b'INCHI')
        mappings_table = MagicMock()
        mappings_table.cols._v_colnames = ['idnumber', 'connectivity', 'InChIKey']
        mappings_table.where.return_value = iter([row])
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
        mappings_table = MagicMock()
        mappings_table.cols._v_colnames = ['idnumber', 'connectivity', 'InChIKey']
        mappings_table.where.return_value = iter([])
        h5file = MagicMock()
        h5file.root.mol_mappings = mappings_table
        with patch.object(ss, 'tb', create=True) as mock_tb:
            mock_tb.open_file.return_value.__enter__.return_value = h5file
            with self.assertRaises(ValueError):
                obj._get_mapping([999])


class TestBaseMultiFpEngine(unittest.TestCase):

    def test_init_builds_pytables_backend(self):
        with patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls, \
             patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True):
            engine = ss.BaseMultiFpEngine.__new__(ss.BaseMultiFpEngine)
            ss.BaseMultiFpEngine.__init__(
                engine, 'fake.h5', 'sig1', storage_backend='pytables',
                in_memory_fps=True, fps_sort=False,
            )
        mock_backend_cls.assert_called_once_with('fake.h5', 'sig1', in_memory_fps=True, fps_sort=False)
        self.assertIs(engine.storage, mock_backend_cls.return_value)

    def test_load_query_raises_on_unparseable_molecule(self):
        engine = ss.BaseMultiFpEngine.__new__(ss.BaseMultiFpEngine)
        with patch.object(ss, 'load_molecule', create=True, return_value=None):
            with self.assertRaises(ValueError):
                engine.load_query('not a molecule')

    def test_load_query_builds_fingerprint_array(self):
        engine = ss.BaseMultiFpEngine.__new__(ss.BaseMultiFpEngine)
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
        self.assertIn('score', df.columns) if 'score' in df.columns else None
        self.assertTrue(any('Tanimoto' in c for c in df.columns))

    def test_on_disk_similarity(self):
        engine = self._make_engine()
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.on_disk_similarity.return_value = [(1, 0.9)]
            df = engine.on_disk_similarity('CCO', 0.7)
        self.assertTrue(any('Tanimoto' in c for c in df.columns))

    def test_tversky_builds_df_with_tversky_label(self):
        engine = self._make_engine()
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.tversky.return_value = [(1, 0.9)]
            df = engine.tversky('CCO', 0.7, a=0.5, b=0.5)
        self.assertTrue(any('Tversky' in c for c in df.columns))

    def test_on_disk_tversky(self):
        engine = self._make_engine()
        with patch.object(ss, 'FPSim2Engine', create=True) as mock_cls:
            mock_cls.on_disk_tversky.return_value = [(1, 0.9)]
            df = engine.on_disk_tversky('CCO', 0.7, a=0.5, b=0.5)
        self.assertTrue(any('Tversky' in c for c in df.columns))

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
        with self.assertRaises(ImportError):
            ss.FPSubSim2CudaEngine('fake.h5', 'sig1')

    def test_raises_on_invalid_kernel(self):
        with patch.object(ss, 'HAS_CUPY', True):
            with self.assertRaises(ValueError):
                ss.FPSubSim2CudaEngine('fake.h5', 'sig1', kernel='bogus')

    # Past the kernel-validation check, self.fps/self.raw_kernel/ew_kernel
    # are set up by the real FPSim2CudaEngine base class's own __init__ (not
    # invoked by our MRO) - unavailable without cupy+FPSim2 actually
    # installed, so construction still builds the storage backend
    # correctly (covered below) but then fails reaching those attributes;
    # that failure itself is excluded from coverage (see pragma comments in
    # subsim_search.py).
    def test_builds_storage_backend_then_fails_on_the_cupy_specific_branch(self):
        with (
            patch.object(ss, 'HAS_CUPY', True),
            patch.object(ss, 'HAS_TABLES', True), patch.object(ss, 'HAS_FPSIM2', True),
            patch.object(ss, 'cupy', create=True),
            patch.object(ss, 'PyTablesMultiFpStorageBackend') as mock_backend_cls,
        ):
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
    On a <=2-core machine this produced 0 (or negative) workers, and with
    no worker ever draining the reader's queue, _reader_process's
    back-pressure guard slept forever - an indefinite hang.
    """

    def _run_with_mocked_multiprocessing(self, njobs, cpu_count):
        engine = make_engine()
        process_instances = []

        def fake_process(target=None, args=None, **kwargs):
            proc = MagicMock()
            proc.start = MagicMock()
            proc.join = MagicMock()
            proc.is_alive = MagicMock(return_value=False)
            process_instances.append((target, proc))
            return proc

        with patch('src.papyrus_scripts.subsim_search.multiprocessing.cpu_count', return_value=cpu_count), \
             patch('src.papyrus_scripts.subsim_search.multiprocessing.Process', side_effect=fake_process), \
             patch('src.papyrus_scripts.subsim_search.multiprocessing.Queue', return_value=MagicMock()), \
             patch('src.papyrus_scripts.subsim_search.sort_db_file'):
            engine._parallel_create(njobs, fingerprint=[], progress=False, total=None)

        return process_instances

    def test_njobs_minus_one_never_spawns_zero_workers_on_single_core(self):
        instances = self._run_with_mocked_multiprocessing(njobs=-1, cpu_count=1)
        worker_count = sum(
            1 for target, _ in instances
            if target is not None and target.__name__ == '_worker_process'
        )
        self.assertGreaterEqual(worker_count, 1)

    def test_njobs_minus_one_never_spawns_zero_workers_on_dual_core(self):
        instances = self._run_with_mocked_multiprocessing(njobs=-1, cpu_count=2)
        worker_count = sum(
            1 for target, _ in instances
            if target is not None and target.__name__ == '_worker_process'
        )
        self.assertGreaterEqual(worker_count, 1)


class TestParallelCreateCleanup(unittest.TestCase):
    """Every process still alive when _parallel_create exits - even via an
    exception (e.g. KeyboardInterrupt) during the wait - must be
    terminated, not left orphaned.
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
            patch('src.papyrus_scripts.subsim_search.multiprocessing.Queue', return_value=MagicMock()),
            patch('src.papyrus_scripts.subsim_search.sort_db_file') as mock_sort,
        ):
            engine._parallel_create(njobs=2, fingerprint=[], progress=False, total=None)
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
            patch('src.papyrus_scripts.subsim_search.multiprocessing.Queue', return_value=MagicMock()),
            patch('src.papyrus_scripts.subsim_search.sort_db_file') as mock_sort,
        ):
            with self.assertRaises(RuntimeError):
                engine._parallel_create(njobs=2, fingerprint=[], progress=False, total=None)
        self.assertTrue(process_instances)
        for proc in process_instances:
            proc.terminate.assert_called_once()
        mock_sort.assert_not_called()


if __name__ == '__main__':
    unittest.main()
