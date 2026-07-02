# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.matchRCSB.get_matches.

Mocks update_rcsb_data, pandas.read_csv and pystow.module entirely: no
network access or real RCSB_data.tsv.xz file is touched. get_matches'
internal merge/groupby logic is pandas-based, so these tests focus on the
data-type boundary (polars DataFrame/LazyFrame input must be accepted and
converted, since the rest of this library is polars-native).
"""

import unittest
from unittest.mock import patch

import pandas as pd
import polars as pl

from src.papyrus_scripts.matchRCSB import get_matches


def make_rcsb_data():
    return pd.DataFrame({
        'InChI_2D': ['InChI=1'],
        'InChI_3D': ['InChI=1_3D'],
        'PDBID_ligand': ['LIG1'],
        'PDBID_protein': ['PDB1'],
        'UniProt_accession': ['P1'],
    })


def make_bioactivity_data():
    return {
        'Activity_ID': ['A1', 'A2'],
        'InChI': ['InChI=1', 'InChI=2'],
        'accession': ['P1', 'P2'],
        'connectivity': ['C1', 'C2'],
    }


class TestGetMatches(unittest.TestCase):

    def setUp(self):
        self.rcsb_data = make_rcsb_data()
        self.update_patch = patch(
            'src.papyrus_scripts.matchRCSB.update_rcsb_data', return_value=None,
        )
        self.read_csv_patch = patch('pandas.read_csv', return_value=self.rcsb_data)
        self.pystow_patch = patch('src.papyrus_scripts.matchRCSB.pystow.module')
        self.update_patch.start()
        self.read_csv_patch.start()
        mock_module = self.pystow_patch.start()
        mock_module.return_value.join.return_value = 'fake_path.tsv'
        self.addCleanup(self.update_patch.stop)
        self.addCleanup(self.read_csv_patch.stop)
        self.addCleanup(self.pystow_patch.stop)

    def test_polars_dataframe_input_does_not_raise(self):
        # Regression test: get_matches' type gate only ever matched pandas
        # objects (isinstance(data, pd.DataFrame)), so a polars DataFrame -
        # what the rest of this library now produces - always fell through
        # to `raise TypeError(...)`.
        df = pl.DataFrame(make_bioactivity_data())
        result = get_matches(df, update=True, verbose=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result['PDBID_ligand'].tolist(), ['LIG1'])

    def test_polars_lazyframe_input_does_not_raise(self):
        lf = pl.DataFrame(make_bioactivity_data()).lazy()
        result = get_matches(lf, update=True, verbose=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result['PDBID_ligand'].tolist(), ['LIG1'])

    def test_pandas_dataframe_input_still_works(self):
        df = pd.DataFrame(make_bioactivity_data())
        result = get_matches(df, update=True, verbose=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result['PDBID_ligand'].tolist(), ['LIG1'])

    def test_invalid_data_type_raises(self):
        with self.assertRaises(TypeError):
            get_matches([1, 2, 3], update=True, verbose=False)


if __name__ == '__main__':
    unittest.main()
