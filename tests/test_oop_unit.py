# -*- coding: utf-8 -*-

"""Lightweight, non-network unit tests for papyrus_scripts.oop.

Unlike tests/test_oop.py (which downloads real Papyrus data end to end),
these tests build a PapyrusDataset directly from a small synthetic
pl.DataFrame via PapyrusDataset._from_data(), to exercise PapyrusDataFilter
without any network access.
"""

import unittest

import polars as pl

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


if __name__ == '__main__':
    unittest.main()
