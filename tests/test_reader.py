# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.reader helper functions that don't require
downloading real Papyrus data.
"""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.papyrus_scripts.reader import _read_custom_protein_descriptors, _read_unirep


class TestReadUnirep(unittest.TestCase):
    """Regression test: _read_unirep used to take a `lazy` flag whose True
    branch did pl.scan_csv(...).collect() - immediately materialising, so
    the "lazy" path never actually streamed anything and was functionally
    identical to the eager pl.read_csv() path, just via a slower detour.
    Simplified to always read eagerly; verify the output is unaffected.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fpath = Path(self._tmpdir.name) / 'unirep.tsv'
        pl.DataFrame({
            'TARGET_NAME': ['P1', 'P2', 'P3'],
            'x0': [0.1, 0.2, 0.3],
        }).write_csv(self.fpath, separator='\t')

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_renames_target_name_column(self):
        result = _read_unirep(self.fpath, schema={}, ids=None)
        self.assertIn('target_id', result.columns)
        self.assertNotIn('TARGET_NAME', result.columns)

    def test_filters_by_ids(self):
        result = _read_unirep(self.fpath, schema={}, ids=['P1', 'P3'])
        self.assertEqual(sorted(result['target_id']), ['P1', 'P3'])

    def test_no_ids_keeps_everything(self):
        result = _read_unirep(self.fpath, schema={}, ids=None)
        self.assertEqual(len(result), 3)


class TestReadCustomProteinDescriptors(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.fpath = Path(self._tmpdir.name) / 'custom.tsv'
        pl.DataFrame({
            'TARGET_NAME': ['P1', 'P2'],
            'feature': [1, 2],
        }).write_csv(self.fpath, separator='\t')

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_renames_target_name_column(self):
        result = _read_custom_protein_descriptors(self.fpath, ids=None)
        self.assertIn('target_id', result.columns)

    def test_filters_by_ids(self):
        result = _read_custom_protein_descriptors(self.fpath, ids=['P1'])
        self.assertEqual(result['target_id'].to_list(), ['P1'])


if __name__ == '__main__':
    unittest.main()
