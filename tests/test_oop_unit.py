# -*- coding: utf-8 -*-

"""Lightweight, non-network unit tests for papyrus_scripts.oop.

Unlike tests/test_oop.py (which downloads real Papyrus data end to end),
these tests build a PapyrusDataset directly from a small synthetic
pl.DataFrame via PapyrusDataset._from_data(), to exercise PapyrusDataFilter
without any network access.
"""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from src.papyrus_scripts.fingerprint import MorganFingerprint
from src.papyrus_scripts.oop import (
    PapyrusDataset,
    PapyrusDescriptorSet,
    PapyrusMoleculeSet,
    PapyrusProteinSet,
)


def make_dataset():
    df = pl.DataFrame({
        'Activity_ID': ['A1', 'A2', 'A3'],
        'connectivity': ['C1', 'C2', 'C3'],
        'Quality': ['High', 'Medium', 'Low'],
        'source': ['chembl', 'chembl;other', 'other'],
        'CID': ['1', '2;3', '4'],
        'AID': ['10', '20;30', '40'],
        'type_IC50': ['1', '1;0', '0'],
        'type_EC50': ['0', '0;1', '1'],
        'type_KD': ['0', '0;0', '0'],
        'type_Ki': ['0', '0;0', '0'],
        'type_other': ['0', '0;0', '0'],
        'relation': ['=', '=;=', '='],
        'pchembl_value': ['6.5', '6.5;7.0', '5.0'],
        'Activity_class': [None, None, None],
        'target_id': ['P1', 'P1', 'P2'],
    })
    protein_data = pl.DataFrame({
        'target_id': ['P1', 'P2'],
        'Organism': ['Homo sapiens (Human)', 'Homo sapiens (Human)'],
        'Classification': ['Enzyme->Kinase', 'Enzyme->Protease'],
    })
    papyrus_params = dict(
        is3d=False, version=None, plusplus=True, chunksize=None,
        source_path=None, num_rows=len(df), download_progress=False,
        keep_original_files=False, disk_margin=0.10,
    )
    return PapyrusDataset._from_data(
        papyrus_bioactivity_data=df,
        papyrus_protein_data=protein_data,
        papyrus_params=papyrus_params,
    )


class TestPapyrusDataFilterKeepSourceAndType(unittest.TestCase):
    """Regression test: PapyrusDataFilter.keep_source/keep_activity_type used to
    forward njobs=/verbose= kwargs to preprocess.keep_source/keep_type, which
    have never accepted them, raising TypeError on every call.
    """

    def setUp(self):
        self.dataset = make_dataset()

    def test_keep_source_does_not_raise(self):
        result = self.dataset.keep_source(source='chembl')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])

    def test_keep_activity_type_does_not_raise(self):
        result = self.dataset.keep_activity_type(activity_types='ic50')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])

    def test_keep_source_via_configured_filter_does_not_raise(self):
        # The class docstring's own documented usage pattern:
        # dataset._filter(njobs=4, progress=True).keep_quality('medium')
        result = self.dataset._filter(njobs=4, progress=True).keep_source(source='chembl')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])

    def test_keep_activity_type_via_configured_filter_does_not_raise(self):
        result = self.dataset._filter(njobs=2, progress=True).keep_activity_type(activity_types='ic50')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])


class TestPapyrusDataFilterProteinClassAndOrganism(unittest.TestCase):
    """keep_protein_class/keep_organism are generated methods requiring
    protein_data injected from self - previously untested outside the
    network-gated tests/test_oop.py.
    """

    def setUp(self):
        self.dataset = make_dataset()

    def test_keep_protein_class_does_not_raise(self):
        result = self.dataset.keep_protein_class({'l2': 'Kinase'})
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])

    def test_keep_protein_class_requires_classes(self):
        # keep_protein_class forces `classes` required despite
        # preprocess.keep_protein_class defaulting it to None.
        with self.assertRaises(TypeError):
            self.dataset.keep_protein_class()

    def test_keep_organism_does_not_raise(self):
        result = self.dataset.keep_organism('Homo sapiens (Human)')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2', 'A3'])


class TestPapyrusDataFilterGenericColumn(unittest.TestCase):
    """contains/not_contains/isin/not_isin are generated methods renamed
    from preprocess.keep_contains/keep_not_contains/keep_match/keep_not_match.
    """

    def setUp(self):
        self.dataset = make_dataset()

    def test_contains_does_not_raise(self):
        result = self.dataset.contains('source', 'chembl')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])

    def test_not_contains_does_not_raise(self):
        result = self.dataset.not_contains('source', 'chembl')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A3'])

    def test_isin_does_not_raise(self):
        result = self.dataset.isin('Quality', ['High'])
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1'])

    def test_not_isin_does_not_raise(self):
        result = self.dataset.not_isin('Quality', ['High'])
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A2', 'A3'])


class TestFPSubSim2EngineFilters(unittest.TestCase):
    """keep_similar_molecules/keep_dissimilar_molecules/keep_substructure_molecules/
    keep_not_substructure_molecules are generated methods with renamed params
    (smiles->molecule_smiles, fp->fingerprint) and injected fpsubsim2_file.
    Mocked at the preprocess.* boundary (real FPSubSim2 search is already
    covered by tests/test_preprocess.py::TestKeepSimilarDissimilarSubstructure);
    FPSubSim2Engine._ensure_loaded is patched to a no-op so no .h5 file or
    network is touched. These tests also verify the generated methods'
    late-binding design: preprocess.<target_name> and MorganFingerprint are
    resolved at call time, so mock.patch here actually takes effect.
    """

    def setUp(self):
        # FPSubSim2Engine.__init__ eagerly constructs a subsim_search.FPSubSim2(),
        # which requires optional deps (tables, FPSim2) not needed for this test.
        self.fpsubsim2_class_patch = patch('src.papyrus_scripts.oop.subsim_search.FPSubSim2')
        self.fpsubsim2_class_patch.start()
        self.addCleanup(self.fpsubsim2_class_patch.stop)

        self.dataset = make_dataset()
        self.ensure_loaded_patch = patch(
            'src.papyrus_scripts.oop.FPSubSim2Engine._ensure_loaded', return_value=None,
        )
        self.ensure_loaded_patch.start()
        self.addCleanup(self.ensure_loaded_patch.stop)

    def test_keep_similar_molecules_delegates_correctly(self):
        called = {}

        def fake_keep_similar(**kwargs):
            called.update(kwargs)
            return self.dataset.papyrus_bioactivity_data

        with patch('src.papyrus_scripts.oop.preprocess.keep_similar', side_effect=fake_keep_similar):
            result = self.dataset.keep_similar_molecules(smiles='CCO', threshold=0.5)

        self.assertEqual(called['molecule_smiles'], 'CCO')
        self.assertEqual(called['threshold'], 0.5)
        self.assertIsInstance(called['fingerprint'], MorganFingerprint)
        self.assertIsInstance(result, PapyrusDataset)

    def test_keep_dissimilar_molecules_delegates_correctly(self):
        called = {}

        def fake_keep_dissimilar(**kwargs):
            called.update(kwargs)
            return self.dataset.papyrus_bioactivity_data

        with patch('src.papyrus_scripts.oop.preprocess.keep_dissimilar', side_effect=fake_keep_dissimilar):
            result = self.dataset.keep_dissimilar_molecules(smiles='CCO', threshold=0.9)

        self.assertEqual(called['molecule_smiles'], 'CCO')
        self.assertEqual(called['threshold'], 0.9)
        self.assertIsInstance(called['fingerprint'], MorganFingerprint)
        self.assertIsInstance(result, PapyrusDataset)

    def test_keep_substructure_molecules_delegates_correctly(self):
        called = {}

        def fake_keep_substructure(**kwargs):
            called.update(kwargs)
            return self.dataset.papyrus_bioactivity_data

        with patch('src.papyrus_scripts.oop.preprocess.keep_substructure', side_effect=fake_keep_substructure):
            result = self.dataset.keep_substructure_molecules(smiles='CCO')

        self.assertEqual(called['molecule_smiles'], 'CCO')
        self.assertNotIn('fingerprint', called)
        self.assertIsInstance(result, PapyrusDataset)

    def test_keep_not_substructure_molecules_delegates_correctly(self):
        called = {}

        def fake_keep_not_substructure(**kwargs):
            called.update(kwargs)
            return self.dataset.papyrus_bioactivity_data

        with patch('src.papyrus_scripts.oop.preprocess.keep_not_substructure', side_effect=fake_keep_not_substructure):
            result = self.dataset.keep_not_substructure_molecules(smiles='CCO')

        self.assertEqual(called['molecule_smiles'], 'CCO')
        self.assertIsInstance(result, PapyrusDataset)


class TestPapyrusDatasetDownloadProgressDefaultsToTrue(unittest.TestCase):
    """download_progress now defaults to True (previously False) on both
    PapyrusDataset.__init__ and PapyrusDataset.from_dataframe, so a bare
    call showing no explicit download_progress= now shows progress bars
    instead of silently downloading with none.
    """

    def test_init_defaults_to_true(self):
        default = inspect.signature(PapyrusDataset.__init__).parameters['download_progress'].default
        self.assertIs(default, True)

    def test_from_dataframe_defaults_to_true(self):
        default = inspect.signature(PapyrusDataset.from_dataframe).parameters['download_progress'].default
        self.assertIs(default, True)

    def test_constructing_without_the_argument_forwards_progress_true(self):
        with (
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', side_effect=[KeyError(), 0]),
            patch('src.papyrus_scripts.oop.reader.read_papyrus', return_value=pl.DataFrame()),
            patch('src.papyrus_scripts.oop.reader.read_protein_set', return_value=pl.DataFrame()),
        ):
            # No download_progress= passed here - relies on the default.
            PapyrusDataset(version='2022.04.2').aggregate()
        self.assertTrue(mock_download.call_args.kwargs['progress'])


class TestPapyrusDatasetDownloadUsesConsistentFolderKey(unittest.TestCase):
    """Regression test: PapyrusDataset.__init__'s auto-download branch used
    to pass pv.version (the canonical new-format string, e.g. '2022.04.2')
    to download_papyrus, while every read afterwards (get_num_rows_in_file,
    read_papyrus, read_protein_set) used the original pv object directly -
    whose pystow_path_key is the *old*-format string (e.g. '05.4') when the
    caller supplied an old-format version. download_papyrus would then
    resolve its own PapyrusVersion from the canonical string and write to a
    different folder ('2022.04.2') than every read looked under ('05.4'),
    surfacing as a KeyError from a missing data_size.json after a
    multi-gigabyte download had already completed. Fixed by passing
    pv.pystow_path_key everywhere instead, so the write and every read agree.
    """

    def _run_init(self, version):
        with (
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            # Raise on the first call (cache miss - triggers the download
            # fallback), succeed on the retry after that download.
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', side_effect=[KeyError(), 0]),
            patch('src.papyrus_scripts.oop.reader.read_papyrus', return_value=pl.DataFrame()),
            patch('src.papyrus_scripts.oop.reader.read_protein_set', return_value=pl.DataFrame()),
        ):
            # Construction alone must not download anything - only
            # aggregate() (here, forcing resolution) does.
            PapyrusDataset(version=version, download_progress=False).aggregate()
        return mock_download

    def test_old_format_version_downloads_under_old_format_folder_key(self):
        mock_download = self._run_init('05.4')
        self.assertEqual(mock_download.call_args.kwargs['version'], '05.4')

    def test_new_format_version_downloads_under_new_format_folder_key(self):
        mock_download = self._run_init('2022.04.2')
        self.assertEqual(mock_download.call_args.kwargs['version'], '2022.04.2')


class TestPapyrusDatasetInitDownloadsOnlyTheMinimum(unittest.TestCase):
    """PapyrusDataset.__init__ used to auto-download everything (both
    stereo variants, structures, every descriptor type) regardless of the
    (is3d, plusplus) actually requested. It must now request only the
    matching bioactivity file (plus the always-needed protein-target file,
    itself unconditionally included by download_papyrus) - not structures,
    not descriptors, not the other stereo/++ combination.
    """

    def _run_init(self, **kwargs):
        with (
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', side_effect=[KeyError(), 0]),
            patch('src.papyrus_scripts.oop.reader.read_papyrus', return_value=pl.DataFrame()),
            patch('src.papyrus_scripts.oop.reader.read_protein_set', return_value=pl.DataFrame()),
        ):
            # Construction alone must not download anything - only
            # aggregate() (here, forcing resolution) does.
            PapyrusDataset(version='2022.04.2', download_progress=False, **kwargs).aggregate()
        return mock_download

    def test_construction_alone_downloads_nothing(self):
        with (
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file') as mock_num_rows,
            patch('src.papyrus_scripts.oop.reader.read_papyrus') as mock_read_papyrus,
            patch('src.papyrus_scripts.oop.reader.read_protein_set') as mock_read_proteins,
        ):
            PapyrusDataset(version='2022.04.2', download_progress=False)
        mock_download.assert_not_called()
        mock_num_rows.assert_not_called()
        mock_read_papyrus.assert_not_called()
        mock_read_proteins.assert_not_called()

    def test_2d_plusplus_requests_only_matching_bioactivities(self):
        mock_download = self._run_init(is3d=False, plusplus=True)
        call = mock_download.call_args.kwargs
        self.assertEqual(call['nostereo'], True)
        self.assertEqual(call['stereo'], False)
        self.assertEqual(call['only_pp'], True)
        self.assertEqual(call['structures'], False)
        self.assertIsNone(call['descriptors'])

    def test_2d_full_requests_only_matching_bioactivities(self):
        mock_download = self._run_init(is3d=False, plusplus=False)
        call = mock_download.call_args.kwargs
        self.assertEqual(call['nostereo'], True)
        self.assertEqual(call['stereo'], False)
        self.assertEqual(call['only_pp'], False)
        self.assertEqual(call['structures'], False)
        self.assertIsNone(call['descriptors'])

    def test_3d_requests_only_matching_bioactivities(self):
        mock_download = self._run_init(is3d=True, plusplus=False)
        call = mock_download.call_args.kwargs
        self.assertEqual(call['nostereo'], False)
        self.assertEqual(call['stereo'], True)
        self.assertEqual(call['structures'], False)
        self.assertIsNone(call['descriptors'])

    def test_no_download_at_all_on_a_full_cache_hit(self):
        with (
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', return_value=0),
            patch('src.papyrus_scripts.oop.reader.read_papyrus', return_value=pl.DataFrame()),
            patch('src.papyrus_scripts.oop.reader.read_protein_set', return_value=pl.DataFrame()),
        ):
            PapyrusDataset(version='2022.04.2', download_progress=False).aggregate()
        mock_download.assert_not_called()


class TestPapyrusDatasetKeepOriginalFiles(unittest.TestCase):
    """PapyrusDataset(..., keep_original_files=...) must be forwarded as
    download_papyrus's keep_xz, both for the initial auto-download and for
    any descriptor file downloaded lazily afterwards - otherwise there is no
    way for an OOP-API caller to keep the .tsv.xz originals that
    download_papyrus converts to Parquet (and deletes) by default.
    """

    def _run_init(self, **kwargs):
        with (
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            # Raise on the first call (cache miss - triggers the download
            # fallback), succeed on the retry after that download.
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', side_effect=[KeyError(), 0]),
            patch('src.papyrus_scripts.oop.reader.read_papyrus', return_value=pl.DataFrame()),
            patch('src.papyrus_scripts.oop.reader.read_protein_set', return_value=pl.DataFrame()),
        ):
            dataset = PapyrusDataset(version='2022.04.2', download_progress=False, **kwargs)
            # Construction alone must not download anything - only
            # aggregate() (here, forcing resolution) does.
            dataset.aggregate()
        return dataset, mock_download

    def test_defaults_to_deleting_originals(self):
        _, mock_download = self._run_init()
        self.assertEqual(mock_download.call_args.kwargs['keep_xz'], False)

    def test_keep_original_files_forwarded_as_keep_xz(self):
        _, mock_download = self._run_init(keep_original_files=True)
        self.assertEqual(mock_download.call_args.kwargs['keep_xz'], True)

    def test_descriptor_fallback_download_forwards_keep_original_files(self):
        dataset, _ = self._run_init(keep_original_files=True)
        with (
            patch(
                'src.papyrus_scripts.oop.reader.read_molecular_descriptors',
                side_effect=[FileNotFoundError, pl.DataFrame()],
            ),
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            patch.object(PapyrusDataset, 'aggregate', return_value=pl.DataFrame({'connectivity': []})),
        ):
            descriptor_set = dataset.molecular_descriptors('mold2')
            mock_download.assert_not_called()  # not until aggregate() is called
            descriptor_set.aggregate()
        self.assertEqual(mock_download.call_args.kwargs['keep_xz'], True)


class TestPapyrusDatasetRemoveUsesConsistentFolderKey(unittest.TestCase):
    """Regression test: PapyrusDataset.remove() passed pv.version (the
    canonical new-format string, e.g. '2024.09.2') to remove_papyrus, so for
    data downloaded under an old-format key (e.g. '05.7') it resolved a
    different, non-existent folder - pystow.module() creates that folder as
    a side effect of resolving the path, leaving an empty '2024.09.2'
    directory behind next to the real '05.7' one instead of touching any
    real data. Fixed by passing pv.pystow_path_key instead, matching
    PapyrusDataset.__init__'s download branch.
    """

    def _run_remove(self, version):
        with patch('src.papyrus_scripts.oop.download.remove_papyrus') as mock_remove:
            PapyrusDataset.remove(
                version=version,
                remove_papyruspp=False, remove_bioactivities=False, remove_proteins=False,
                remove_nostereo=False, remove_stereo=False, remove_structures=False,
                remove_descriptors=[], remove_other_files=False,
                remove_version_root=False, remove_papyrus_root=False,
                force=True, progress=False,
            )
        return mock_remove

    def test_old_format_version_resolves_old_format_folder_key(self):
        mock_remove = self._run_remove('05.4')
        self.assertEqual(mock_remove.call_args.kwargs['version'], '05.4')

    def test_new_format_version_resolves_new_format_folder_key(self):
        mock_remove = self._run_remove('2022.04.2')
        self.assertEqual(mock_remove.call_args.kwargs['version'], '2022.04.2')


class TestDerivedSetReprShowsRealCount(unittest.TestCase):
    """Regression test: PapyrusMoleculeSet/PapyrusProteinSet.__repr__ checked
    isinstance(self.data, pd.DataFrame), but self.data is always a polars
    DataFrame (never pandas) for these two classes - the check never matched,
    so __repr__ always claimed "<iterator of X>" even for a concrete,
    already-materialised DataFrame.
    """

    def test_molecule_set_repr_shows_count_for_materialized_dataframe(self):
        df = pl.DataFrame({'connectivity': ['C1', 'C2'], 'mol': [None, None]})
        fake_dataset = MagicMock()
        fake_dataset.papyrus_params = {
            'is3d': False, 'version': None, 'plusplus': True,
            'chunksize': None, 'source_path': None,
            'download_progress': False, 'keep_original_files': False,
        }
        # PapyrusMoleculeSet is purely lazy - construction never touches
        # self._dataset, so .data is set directly here to simulate an
        # already-materialised set without exercising _ensure_loaded.
        mset = PapyrusMoleculeSet(fake_dataset)
        mset.data = df
        self.assertEqual(repr(mset), 'PapyrusMoleculeSet<2 molecules>')

    def test_molecule_set_repr_shows_not_yet_materialised_before_aggregate(self):
        fake_dataset = MagicMock()
        fake_dataset.papyrus_params = {
            'is3d': False, 'version': None, 'plusplus': True,
            'chunksize': None, 'source_path': None,
            'download_progress': False, 'keep_original_files': False,
        }
        mset = PapyrusMoleculeSet(fake_dataset)
        self.assertEqual(repr(mset), 'PapyrusMoleculeSet<not yet materialised>')
        fake_dataset.aggregate.assert_not_called()

    def test_protein_set_repr_shows_count_for_materialized_dataframe(self):
        df = pl.DataFrame({'target_id': ['P1', 'P2']})
        params = {'is3d': False, 'version': None, 'plusplus': True,
                  'chunksize': None, 'source_path': None}
        pset = PapyrusProteinSet(df, params, num_proteins=2)
        self.assertEqual(repr(pset), 'PapyrusProteinSet<2 proteins>')


class TestMoleculesAndMolecularDescriptorsAreLazy(unittest.TestCase):
    """PapyrusDataset.molecules()/.molecular_descriptors() used to download
    (or read) the structure/descriptor file - and materialise this
    dataset's own bioactivity data to compute ids - as soon as they were
    called. Both must now be purely lazy constructors: nothing is read or
    downloaded until aggregate()/agg()/consume_chunks()/to_dataframe() is
    called on the object they return.
    """

    def setUp(self):
        self.dataset = make_dataset()
        # A real PapyrusVersion isn't needed for these tests - only its
        # pystow_path_key, used when building the download-fallback call.
        fake_version = MagicMock()
        fake_version.pystow_path_key = '2022.04.2'
        self.dataset.papyrus_params['version'] = fake_version

    def test_molecules_touches_nothing_until_aggregate(self):
        with (
            patch('src.papyrus_scripts.oop.reader.read_molecular_structures') as mock_read,
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
        ):
            mol_set = self.dataset.molecules()
            mock_read.assert_not_called()
            mock_download.assert_not_called()
        self.assertIsInstance(mol_set, PapyrusMoleculeSet)
        self.assertIsNone(mol_set.data)

    def test_molecular_descriptors_touches_nothing_until_aggregate(self):
        with (
            patch('src.papyrus_scripts.oop.reader.read_molecular_descriptors') as mock_read,
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
        ):
            desc_set = self.dataset.molecular_descriptors('mold2')
            mock_read.assert_not_called()
            mock_download.assert_not_called()
        self.assertIsInstance(desc_set, PapyrusDescriptorSet)
        self.assertIsNone(desc_set.data)

    def test_molecules_aggregate_reads_without_downloading_on_a_cache_hit(self):
        mol_set = self.dataset.molecules()
        structures = pl.DataFrame({'connectivity': [], 'mol': []})
        with (
            patch(
                'src.papyrus_scripts.oop.reader.read_molecular_structures',
                return_value=structures,
            ) as mock_read,
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', return_value=0),
        ):
            result = mol_set.aggregate()
        mock_read.assert_called_once()
        mock_download.assert_not_called()
        self.assertIs(result, structures)

    def test_molecules_aggregate_downloads_once_on_a_cache_miss(self):
        mol_set = self.dataset.molecules()
        structures = pl.DataFrame({'connectivity': [], 'mol': []})
        with (
            patch(
                'src.papyrus_scripts.oop.reader.read_molecular_structures',
                side_effect=[FileNotFoundError, structures],
            ) as mock_read,
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', return_value=0),
        ):
            result = mol_set.aggregate()
        self.assertEqual(mock_read.call_count, 2)
        mock_download.assert_called_once()
        self.assertEqual(mock_download.call_args.kwargs['structures'], True)
        self.assertIsNone(mock_download.call_args.kwargs['descriptors'])
        self.assertIs(result, structures)

    def test_molecules_aggregate_only_fetches_once_across_repeated_calls(self):
        mol_set = self.dataset.molecules()
        structures = pl.DataFrame({'connectivity': [], 'mol': []})
        with (
            patch(
                'src.papyrus_scripts.oop.reader.read_molecular_structures',
                return_value=structures,
            ) as mock_read,
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', return_value=0),
        ):
            mol_set.aggregate()
            mol_set.agg()
            mol_set.consume_chunks()
            mol_set.to_dataframe()
        mock_read.assert_called_once()

    def test_molecular_descriptors_aggregate_downloads_once_on_a_cache_miss(self):
        desc_set = self.dataset.molecular_descriptors('mold2')
        descriptors = pl.DataFrame({'connectivity': [], 'CDDD_1': []})
        with (
            patch(
                'src.papyrus_scripts.oop.reader.read_molecular_descriptors',
                side_effect=[FileNotFoundError, descriptors],
            ) as mock_read,
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
        ):
            result = desc_set.aggregate()
        self.assertEqual(mock_read.call_count, 2)
        mock_download.assert_called_once()
        self.assertEqual(mock_download.call_args.kwargs['descriptors'], 'mold2')
        self.assertEqual(mock_download.call_args.kwargs['structures'], False)
        self.assertIs(result, descriptors)


class TestFullChainDefersEverythingToAggregate(unittest.TestCase):
    """Regression test for the reported scenario:
    PapyrusDataset(...).keep_quality(...).keep_accession(...)
    .molecular_descriptors(...).agg() fired two download calls - one
    eagerly at construction (for bioactivities), one at .agg() (for the
    descriptor file) - even though nothing before .agg() should touch the
    network at all. Both must now be deferred to .agg() (or an
    equivalent): building the chain must not download or read anything.

    A later regression (still open when this docstring was written) then
    had .agg() itself fire two *separate* download-and-convert cycles back
    to back - one for bioactivities (triggered by materialising the
    bioactivity stream), one for the descriptor file (triggered once
    reading it failed *after* that) - each spinning up its own
    tqdm/multiprocessing conversion pipeline, which is what actually
    produced the garbled/duplicated progress bars reported alongside this.
    Since .molecular_descriptors(...) is called before .agg() anywhere in
    the chain, .agg() must instead resolve to exactly *one* combined
    download_papyrus() call covering bioactivities, proteins and every
    descriptor set/structures requested in the chain so far.
    """

    def setUp(self):
        self.bioactivity_df = pl.DataFrame({
            'connectivity': ['C1', 'C2'],
            'target_id': ['P00533_WT', 'P00534_WT'],
            'Quality': ['High', 'Medium'],
            'Activity_ID': ['A1', 'A2'], 'source': ['chembl', 'chembl'],
            'CID': ['1', '2'], 'AID': ['10', '20'],
            'type_IC50': ['1', '1'], 'type_EC50': ['0', '0'],
            'type_KD': ['0', '0'], 'type_Ki': ['0', '0'], 'type_other': ['0', '0'],
            'relation': ['=', '='], 'pchembl_value': ['6.5', '7.0'],
            'Activity_class': [None, None],
        })
        self.mold2_df = pl.DataFrame({'connectivity': ['C1'], 'mold2_1': [1.0]})

        patches = {
            'download': patch('src.papyrus_scripts.oop.download.download_papyrus'),
            'num_rows': patch(
                'src.papyrus_scripts.oop.IO.get_num_rows_in_file', side_effect=[KeyError(), 2],
            ),
            'read_papyrus': patch(
                'src.papyrus_scripts.oop.reader.read_papyrus', return_value=self.bioactivity_df,
            ),
            'read_proteins': patch(
                'src.papyrus_scripts.oop.reader.read_protein_set',
                return_value=pl.DataFrame({'target_id': []}),
            ),
            'read_descriptors': patch(
                'src.papyrus_scripts.oop.reader.read_molecular_descriptors',
                return_value=self.mold2_df,
            ),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

        # Real network/read calls are mocked above, so a real (already-
        # known-locally) version string is fine here - it never touches disk.
        self.dataset = PapyrusDataset(
            version='2022.04.2', plusplus=False, download_progress=True,
        )

    def test_building_the_chain_touches_nothing(self):
        self.dataset.keep_quality('high').keep_accession('P00533').molecular_descriptors('mold2')
        self.mocks['download'].assert_not_called()
        self.mocks['read_papyrus'].assert_not_called()
        self.mocks['read_proteins'].assert_not_called()
        self.mocks['read_descriptors'].assert_not_called()

    def test_agg_downloads_exactly_once_with_descriptors_included(self):
        result = (
            self.dataset
            .keep_quality('high')
            .keep_accession('P00533')
            .molecular_descriptors('mold2')
            .agg()
        )
        self.mocks['download'].assert_called_once()
        call = self.mocks['download'].call_args
        self.assertEqual(call.kwargs['descriptors'], ['mold2'])
        self.assertFalse(call.kwargs['structures'])
        self.assertIs(result, self.mold2_df)

    def test_agg_downloads_once_even_when_bioactivity_already_present(self):
        # Bioactivity/proteins are already on disk (no KeyError this time),
        # but the registered descriptor set is missing - still exactly one
        # combined download, not zero-then-one.
        self.mocks['num_rows'].side_effect = None
        self.mocks['num_rows'].return_value = 2
        with patch(
            'src.papyrus_scripts.oop.reader.molecular_descriptors_available',
            return_value=False,
        ):
            result = (
                self.dataset
                .keep_quality('high')
                .keep_accession('P00533')
                .molecular_descriptors('mold2')
                .agg()
            )
        self.mocks['download'].assert_called_once()
        self.assertEqual(self.mocks['download'].call_args.kwargs['descriptors'], ['mold2'])
        self.assertIs(result, self.mold2_df)

    def test_agg_does_not_download_when_everything_already_present(self):
        self.mocks['num_rows'].side_effect = None
        self.mocks['num_rows'].return_value = 2
        with patch(
            'src.papyrus_scripts.oop.reader.molecular_descriptors_available',
            return_value=True,
        ):
            result = (
                self.dataset
                .keep_quality('high')
                .keep_accession('P00533')
                .molecular_descriptors('mold2')
                .agg()
            )
        self.mocks['download'].assert_not_called()
        self.assertIs(result, self.mold2_df)


class TestAggTriggersDownloadForNeverBeforeSeenVersion(unittest.TestCase):
    """Regression test: on a machine that already has *some* Papyrus data
    downloaded (so utils.IO.get_downloaded_versions returns non-empty and
    process_data_version's OSError branch - "no Papyrus data found locally
    at all" - is never reached) but not the specific version requested,
    process_data_version raises ValueError instead ("Version ... is not
    available locally"). reader.read_papyrus (via _resolve_version) raises
    that same ValueError - which _PapyrusSource._ensure_loaded's except
    clause used to not catch at all, crashing .agg() outright the first
    time any given version was ever requested, instead of downloading it.
    """

    def setUp(self):
        self.bioactivity_df = pl.DataFrame({
            'connectivity': ['C1'],
            'target_id': ['P00533_WT'],
            'Quality': ['High'],
            'Activity_ID': ['A1'], 'source': ['chembl'],
            'CID': ['1'], 'AID': ['10'],
            'type_IC50': ['1'], 'type_EC50': ['0'],
            'type_KD': ['0'], 'type_Ki': ['0'], 'type_other': ['0'],
            'relation': ['='], 'pchembl_value': ['6.5'],
            'Activity_class': [None],
        })
        patches = {
            'download': patch('src.papyrus_scripts.oop.download.download_papyrus'),
            'num_rows': patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', return_value=1),
            'read_papyrus': patch(
                'src.papyrus_scripts.oop.reader.read_papyrus',
                side_effect=[
                    ValueError(
                        "Version '2022.04.2' is not available locally.\n"
                        "Downloaded versions: [2022.11.4, 2024.09.2]"
                    ),
                    self.bioactivity_df,
                ],
            ),
            'read_proteins': patch(
                'src.papyrus_scripts.oop.reader.read_protein_set',
                return_value=pl.DataFrame({'target_id': []}),
            ),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)
        self.dataset = PapyrusDataset(
            version='2022.04.2', plusplus=False, download_progress=False,
        )

    def test_agg_downloads_instead_of_crashing(self):
        result = self.dataset.keep_quality('high').agg()
        self.mocks['download'].assert_called_once()
        self.assertEqual(self.mocks['read_papyrus'].call_count, 2)
        self.assertEqual(result['connectivity'].to_list(), ['C1'])


if __name__ == '__main__':
    unittest.main()
