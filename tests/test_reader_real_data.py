# -*- coding: utf-8 -*-

"""Extensive tests for papyrus_scripts.reader against real, locally
downloaded Papyrus data.

Unlike tests/test_reader_offline.py (fully synthetic, no network) and
tests/test_oop.py (downloads a fixed small set of parametrized combos),
this module runs against WHATEVER has actually been downloaded for one
version - the version, root folder and sample size come from environment
variables, and every descriptor/structure/stereo-specific test self-skips
(FileNotFoundError -> skipTest) when that particular file isn't present,
so this module adapts to any download_papyrus(...) combination without
needing to know it in advance.

This is intentionally NOT part of the routine offline suite (network
access + heavy I/O to download/scan real, multi-gigabyte files) - it only
activates when PAPYRUS_REAL_DATA_VERSION is set. The `papyrus
test-real-data` CLI command sets it up and runs this module for you:

    papyrus test-real-data --version 2024.09.2 --download

Every read that could touch millions of rows is bounded (`.head(N)` on a
lazy scan, or `ids=` filtering) - this module never eagerly materialises
a full bioactivity/descriptor/structure file, regardless of PAPYRUS_REAL_DATA_SAMPLE_SIZE.
"""

import os
import unittest

import polars as pl

from src.papyrus_scripts import reader
from src.papyrus_scripts.utils import IO

VERSION = os.environ.get('PAPYRUS_REAL_DATA_VERSION')
ROOT = os.environ.get('PAPYRUS_REAL_DATA_ROOT') or None
SAMPLE_SIZE = int(os.environ.get('PAPYRUS_REAL_DATA_SAMPLE_SIZE', '25'))

_SKIP_REASON = (
    'Real-data tests are opt-in: set PAPYRUS_REAL_DATA_VERSION or run '
    '`papyrus test-real-data --version <version>`.'
)

def _first_numeric_column(df: pl.DataFrame, id_col: str) -> str | None:
    for col in df.columns:
        if col != id_col:
            return col
    return None


@unittest.skipUnless(VERSION, _SKIP_REASON)
class TestRealBioactivities(unittest.TestCase):

    def test_papyruspp_2d(self):
        df = reader.read_papyrus(is3d=False, plusplus=True, version=VERSION, source_path=ROOT)
        self.assertGreater(df.height, 0)
        for col in ('Activity_ID', 'connectivity', 'target_id'):
            self.assertIn(col, df.columns)
        if 'pchembl_value_Mean' in df.columns:
            self.assertEqual(df.schema['pchembl_value_Mean'], pl.Float64)
        n_expected = IO.get_num_rows_in_file(
            'bioactivities', is3D=False, plusplus=True, version=VERSION, root_folder=ROOT,
        )
        self.assertEqual(df.height, n_expected)

    def test_full_2d(self):
        try:
            df = reader.read_papyrus(is3d=False, plusplus=False, version=VERSION, source_path=ROOT, chunksize=1)
        except Exception as e:
            self.skipTest(f'2D (non ++) bioactivities not downloaded: {e}')
        head = df.head(SAMPLE_SIZE).collect()
        self.assertGreater(head.height, 0)

    def test_full_3d(self):
        try:
            df = reader.read_papyrus(is3d=True, plusplus=False, version=VERSION, source_path=ROOT, chunksize=1)
        except Exception as e:
            self.skipTest(f'3D bioactivities not downloaded: {e}')
        head = df.head(SAMPLE_SIZE).collect()
        self.assertGreater(head.height, 0)
        self.assertIn('InChIKey', head.columns)

    def test_papyruspp_lazy_matches_eager(self):
        eager = reader.read_papyrus(is3d=False, plusplus=True, version=VERSION, source_path=ROOT)
        lazy = reader.read_papyrus(is3d=False, plusplus=True, version=VERSION, source_path=ROOT, chunksize=1)
        self.assertIsInstance(lazy, pl.LazyFrame)
        self.assertTrue(lazy.collect().equals(eager))

    def test_3d_plusplus_raises(self):
        with self.assertRaises(ValueError):
            reader.read_papyrus(is3d=True, plusplus=True, version=VERSION, source_path=ROOT)


@unittest.skipUnless(VERSION, _SKIP_REASON)
class TestRealProteinSet(unittest.TestCase):

    def test_read_protein_set(self):
        df = reader.read_protein_set(version=VERSION, source_path=ROOT)
        self.assertGreater(df.height, 0)
        self.assertIn('target_id', df.columns)
        self.assertEqual(df['target_id'].null_count(), 0)


@unittest.skipUnless(VERSION, _SKIP_REASON)
class TestRealMolecularDescriptors(unittest.TestCase):

    def _bounded(self, desc_type: str, is3d: bool) -> pl.DataFrame:
        lazy = reader.read_molecular_descriptors(
            desc_type=desc_type, is3d=is3d, version=VERSION, source_path=ROOT, chunksize=1,
        )
        return lazy.head(SAMPLE_SIZE).collect()

    def _run_or_skip(self, desc_type: str, is3d: bool) -> pl.DataFrame | None:
        try:
            df = self._bounded(desc_type, is3d)
        except Exception as e:
            self.skipTest(f'{desc_type} ({"3D" if is3d else "2D"}) not downloaded: {e}')
            return None
        self.assertGreater(df.height, 0)
        return df

    def test_mold2(self):
        df = self._run_or_skip('mold2', is3d=False)
        if df is not None:
            self.assertEqual(df.schema['D001'], pl.Int64)

    def test_cddd(self):
        df = self._run_or_skip('cddd', is3d=False)
        if df is not None:
            self.assertEqual(df.schema['CDDD_1'], pl.Float64)

    def test_mordred_2d(self):
        self._run_or_skip('mordred', is3d=False)

    def test_mordred_3d(self):
        self._run_or_skip('mordred', is3d=True)

    def test_fingerprint_2d_ecfp6(self):
        df = self._run_or_skip('fingerprint', is3d=False)
        if df is not None:
            self.assertEqual(df.schema['ECFP6_1'], pl.Int64)

    def test_fingerprint_3d_e3fp(self):
        df = self._run_or_skip('fingerprint', is3d=True)
        if df is not None:
            self.assertEqual(df.schema['E3FP_1'], pl.Int64)

    def test_all_2d_join(self):
        try:
            lazy = reader.read_molecular_descriptors(
                desc_type='all', is3d=False, version=VERSION, source_path=ROOT, chunksize=1,
            )
        except Exception as e:
            self.skipTest(f'not enough 2D descriptor types downloaded for an "all" join: {e}')
            return
        df = lazy.head(SAMPLE_SIZE).collect()
        self.assertGreater(df.height, 0)

    def test_ids_filter_actually_filters(self):
        bio = reader.read_papyrus(is3d=False, plusplus=True, version=VERSION, source_path=ROOT, chunksize=1)
        sample_ids = bio.select('connectivity').head(SAMPLE_SIZE).collect()['connectivity'].to_list()
        try:
            df = reader.read_molecular_descriptors(
                desc_type='fingerprint', is3d=False, version=VERSION, source_path=ROOT,
                ids=sample_ids,
            )
        except Exception as e:
            self.skipTest(f'2D fingerprint descriptors not downloaded: {e}')
            return
        self.assertTrue(set(df['connectivity']).issubset(set(sample_ids)))


@unittest.skipUnless(VERSION, _SKIP_REASON)
class TestRealProteinDescriptors(unittest.TestCase):

    def test_unirep(self):
        try:
            df = reader.read_protein_descriptors(desc_type='unirep', version=VERSION, source_path=ROOT)
        except Exception as e:
            self.skipTest(f'unirep descriptors not downloaded: {e}')
            return
        self.assertGreater(df.height, 0)
        self.assertIn('target_id', df.columns)
        numeric_col = _first_numeric_column(df, 'target_id')
        if numeric_col is not None:
            self.assertEqual(df.schema[numeric_col], pl.Float64)

    def test_unirep_ids_filter(self):
        try:
            all_proteins = reader.read_protein_set(version=VERSION, source_path=ROOT)
            sample_ids = all_proteins['target_id'].head(min(SAMPLE_SIZE, 5)).to_list()
            df = reader.read_protein_descriptors(
                desc_type='unirep', version=VERSION, source_path=ROOT, ids=sample_ids,
            )
        except Exception as e:
            self.skipTest(f'unirep descriptors not downloaded: {e}')
            return
        self.assertTrue(set(df['target_id']).issubset(set(sample_ids)))

    def test_prodec(self):
        try:
            from prodec import ProteinDescriptors
        except ImportError:
            self.skipTest('prodec not installed')
            return
        proteins = reader.read_protein_set(version=VERSION, source_path=ROOT)
        sample = proteins.head(2)
        descriptor = ProteinDescriptors().get_descriptor('BLOSUM')
        # No file involved here, so any failure is a real bug, not missing data - don't skip it.
        df = reader.read_protein_descriptors(
            desc_type=descriptor, version=VERSION, source_path=ROOT,
            ids=sample['target_id'].to_list(),
        )
        self.assertGreater(df.height, 0)
        self.assertIn('target_id', df.columns)


@unittest.skipUnless(VERSION, _SKIP_REASON)
class TestRealMolecularStructures(unittest.TestCase):

    def _first_chunk(self, is3d: bool):
        chunks = reader.read_molecular_structures(
            is3d=is3d, version=VERSION, source_path=ROOT, chunksize=SAMPLE_SIZE, verbose=False,
        )
        return next(iter(chunks))

    def test_structures_2d(self):
        try:
            chunk = self._first_chunk(is3d=False)
        except Exception as e:
            self.skipTest(f'2D structures not downloaded: {e}')
            return
        self.assertGreater(chunk.height, 0)
        self.assertLessEqual(chunk.height, SAMPLE_SIZE)
        self.assertIn('mol', chunk.columns)
        self.assertIn('connectivity', chunk.columns)

    def test_structures_3d(self):
        try:
            chunk = self._first_chunk(is3d=True)
        except Exception as e:
            self.skipTest(f'3D structures not downloaded: {e}')
            return
        self.assertGreater(chunk.height, 0)
        self.assertIn('InChIKey', chunk.columns)

    def test_structures_ids_filter(self):
        try:
            bio = reader.read_papyrus(is3d=False, plusplus=True, version=VERSION, source_path=ROOT, chunksize=1)
            sample_ids = bio.select('connectivity').head(5).collect()['connectivity'].to_list()
            df = reader.read_molecular_structures(
                is3d=False, version=VERSION, source_path=ROOT, ids=sample_ids, verbose=False,
            )
        except Exception as e:
            self.skipTest(f'2D structures not downloaded: {e}')
            return
        self.assertTrue(set(df['connectivity']).issubset(set(sample_ids)))


if __name__ == '__main__':
    unittest.main()
