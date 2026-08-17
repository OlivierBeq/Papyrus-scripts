# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.matchRCSB.get_matches.

Mocks update_rcsb_data, polars.read_csv and pystow.module - no network or
real RCSB_data.tsv.xz file touched. Covers the data-type boundary (pandas/
polars DataFrame/LazyFrame input) and correctness of the match/aggregate
output (matching runs in polars, always returns pandas).
"""

import unittest
from unittest.mock import patch

import pandas as pd
import polars as pl

from src.papyrus_scripts.matchRCSB import get_matches


def make_rcsb_data():
    return pl.DataFrame({
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
        self.read_csv_patch = patch('polars.read_csv', return_value=self.rcsb_data)
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


def make_multi_match_rcsb_data():
    # Two RCSB rows for the same (InChI_2D, UniProt_accession) pair - two
    # PDB structures matching the same ligand/protein - to exercise the
    # ';'.join aggregation (every other column keeps its first value).
    return pl.DataFrame({
        'InChI_2D': ['InChI=1', 'InChI=1'],
        'InChI_3D': ['InChI=1_3D', 'InChI=1_3D'],
        'PDBID_ligand': ['LIG1', 'LIG1'],
        'PDBID_protein': ['PDB1', 'PDB2'],
        'UniProt_accession': ['P1', 'P1'],
    })


class TestGetMatchesAggregation(unittest.TestCase):
    """Golden-output tests for get_matches' merge + aggregate step: one
    Activity_ID matching several RCSB rows must collapse to one row, with
    PDBID_protein values ';'-joined and every other column keeping its
    first value.
    """

    def setUp(self):
        self.rcsb_data = make_multi_match_rcsb_data()
        self.update_patch = patch(
            'src.papyrus_scripts.matchRCSB.update_rcsb_data', return_value=None,
        )
        self.read_csv_patch = patch('polars.read_csv', return_value=self.rcsb_data)
        self.pystow_patch = patch('src.papyrus_scripts.matchRCSB.pystow.module')
        self.update_patch.start()
        self.read_csv_patch.start()
        mock_module = self.pystow_patch.start()
        mock_module.return_value.join.return_value = 'fake_path.tsv'
        self.addCleanup(self.update_patch.stop)
        self.addCleanup(self.read_csv_patch.stop)
        self.addCleanup(self.pystow_patch.stop)

    def _run(self, data):
        return get_matches(data, update=True, verbose=False)

    def test_multiple_pdb_matches_are_joined_by_semicolon(self):
        result = self._run(pd.DataFrame(make_bioactivity_data()))
        self.assertEqual(list(result.index), ['A1'])
        self.assertEqual(result.loc['A1', 'PDBID_protein'], 'PDB1;PDB2')

    def test_other_columns_keep_first_value(self):
        result = self._run(pd.DataFrame(make_bioactivity_data()))
        self.assertEqual(result.loc['A1', 'PDBID_ligand'], 'LIG1')
        self.assertEqual(result.loc['A1', 'connectivity'], 'C1')

    def test_unmatched_activity_is_dropped(self):
        result = self._run(pd.DataFrame(make_bioactivity_data()))
        self.assertNotIn('A2', result.index)

    def test_polars_input_gives_the_same_result_as_pandas(self):
        pandas_result = self._run(pd.DataFrame(make_bioactivity_data()))
        polars_result = self._run(pl.DataFrame(make_bioactivity_data()))
        pd.testing.assert_frame_equal(
            pandas_result.sort_index(axis=1), polars_result.sort_index(axis=1),
        )


if __name__ == '__main__':
    unittest.main()
