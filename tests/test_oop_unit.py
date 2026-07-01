# -*- coding: utf-8 -*-

"""Lightweight, non-network unit tests for papyrus_scripts.oop.

Unlike tests/test_oop.py (which downloads real Papyrus data end to end),
these tests build a PapyrusDataset directly from a small synthetic
pl.DataFrame via PapyrusDataset._from_data(), to exercise PapyrusDataFilter
without any network access.
"""

import unittest
from unittest.mock import patch

import polars as pl

from src.papyrus_scripts.fingerprint import MorganFingerprint
from src.papyrus_scripts.oop import PapyrusDataset


def make_dataset():
    df = pl.DataFrame({
        'Activity_ID': ['A1', 'A2', 'A3'],
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


if __name__ == '__main__':
    unittest.main()
