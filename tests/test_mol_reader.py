# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.utils.mol_reader.

All fixtures are small, real molecule files written to a temp directory
(no downloads, no mocking of RDKit): a couple of SMILES/SDF records are
enough to exercise the supplier/compression/format-detection logic.
"""

import bz2
import gzip
import lzma
import os
import tempfile
import unittest

from rdkit import Chem

from src.papyrus_scripts.utils.mol_reader import MolSupplier, ForwardSmilesMolSupplier


def make_supplier_probe():
    """A MolSupplier instance usable to call pure helper methods on."""
    return MolSupplier(supplier=iter([]))


class TestCompressionDetection(unittest.TestCase):

    def setUp(self):
        self.ms = make_supplier_probe()

    def test_detects_lzma(self):
        self.assertEqual(self.ms._get_compression('foo.sd.xz'), ('lzma', 'foo.sd'))

    def test_detects_zlib(self):
        self.assertEqual(self.ms._get_compression('foo.sd.gz'), ('zlib', 'foo.sd'))

    def test_detects_bz2(self):
        self.assertEqual(self.ms._get_compression('foo.sd.bz2'), ('bz2', 'foo.sd'))

    def test_no_compression(self):
        self.assertEqual(self.ms._get_compression('foo.sd'), (None, 'foo.sd'))

    def test_compression_handler_mapping(self):
        self.assertIs(self.ms._get_compression_handler('lzma'), lzma.open)
        self.assertIs(self.ms._get_compression_handler('zlib'), gzip.open)
        self.assertIs(self.ms._get_compression_handler('bz2'), bz2.open)
        self.assertIs(self.ms._get_compression_handler(None), open)

    def test_compression_handler_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.ms._get_compression_handler('bogus')


class TestFormatDetection(unittest.TestCase):

    def setUp(self):
        self.ms = make_supplier_probe()

    def test_smi(self):
        self.assertEqual(self.ms._get_format('foo.smi'), 'smi')

    def test_sd_and_sdf(self):
        self.assertEqual(self.ms._get_format('foo.sd'), 'sd')
        self.assertEqual(self.ms._get_format('foo.sdf'), 'sd')

    def test_mol2(self):
        self.assertEqual(self.ms._get_format('foo.mol2'), 'mol2')

    def test_mol(self):
        self.assertEqual(self.ms._get_format('foo.mol'), 'mol')

    def test_mae(self):
        self.assertEqual(self.ms._get_format('foo.mae'), 'mae')


class TestConstructionErrors(unittest.TestCase):

    def test_requires_source_or_supplier(self):
        with self.assertRaises(ValueError):
            MolSupplier()

    def test_invalid_compression_raises(self):
        with self.assertRaises(ValueError):
            MolSupplier('foo.sd', compression='bogus')

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            MolSupplier('foo.sd', format='bogus')

    def test_file_like_object_requires_format(self):
        import io
        with self.assertRaises(ValueError):
            MolSupplier(io.StringIO('CCO'))

    def test_invalid_source_type_raises(self):
        with self.assertRaises(ValueError):
            MolSupplier(source=123)


class SmallSDFFixture(unittest.TestCase):
    """Base class writing a small, real 2-molecule SDF fixture to a temp dir."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.sd_path = os.path.join(self._tmpdir.name, 'mols.sd')
        writer = Chem.SDWriter(self.sd_path)
        for smi in ('CCO', 'c1ccccc1'):
            writer.write(Chem.MolFromSmiles(smi))
        writer.close()

    def tearDown(self):
        self._tmpdir.cleanup()


class TestMolSupplierSDFPlain(SmallSDFFixture):

    def test_reads_uncompressed_sdf(self):
        with MolSupplier(self.sd_path) as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 2)
        smiles = sorted(Chem.MolToSmiles(m) for _, m in mols)
        self.assertEqual(smiles, sorted(['CCO', 'c1ccccc1']))

    def test_molecule_ids_start_at_zero_by_default(self):
        with MolSupplier(self.sd_path) as ms:
            ids = [i for i, _ in ms]
        self.assertEqual(ids, [0, 1])

    def test_start_id_offset(self):
        with MolSupplier(self.sd_path, start_id=5, show_progress=False) as ms:
            ids = [i for i, _ in ms]
        self.assertEqual(ids, [5, 6])

    def test_explicit_format_bypasses_autodetection(self):
        with MolSupplier(self.sd_path, format='sd') as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 2)


class TestMolSupplierSDFCompressed(SmallSDFFixture):

    def _compress(self, opener, suffix):
        with open(self.sd_path, 'rb') as fin:
            data = fin.read()
        path = self.sd_path + suffix
        with opener(path, 'wb') as fout:
            fout.write(data)
        return path

    def test_reads_lzma_compressed_sdf_autodetected(self):
        path = self._compress(lzma.open, '.xz')
        with MolSupplier(path) as ms:
            self.assertEqual(len(list(ms)), 2)

    def test_reads_gzip_compressed_sdf_autodetected(self):
        path = self._compress(gzip.open, '.gz')
        with MolSupplier(path) as ms:
            self.assertEqual(len(list(ms)), 2)

    def test_reads_bz2_compressed_sdf_autodetected(self):
        path = self._compress(bz2.open, '.bz2')
        with MolSupplier(path) as ms:
            self.assertEqual(len(list(ms)), 2)

    def test_reads_with_explicit_compression_and_no_explicit_format(self):
        # Regression test: passing `compression` explicitly (format left to be
        # auto-detected) used to crash with AttributeError because
        # self._trunc_filename was only ever set on the auto-detect path.
        path = self._compress(gzip.open, '.gz')
        with MolSupplier(path, compression='zlib') as ms:
            self.assertEqual(len(list(ms)), 2)


class TestMolSupplierSMI(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.smi_path = os.path.join(self._tmpdir.name, 'mols.smi')
        with open(self.smi_path, 'w') as f:
            f.write('SMILES\tName\n')
            f.write('CCO\tethanol\n')
            f.write('c1ccccc1\tbenzene\n')

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_reads_tab_delimited_smi_with_title_line(self):
        # Regression test: ForwardSmilesMolSupplier used to pass the record
        # separator ('\n') as RDKit's field delimiter instead of the actual
        # column delimiter (e.g. '\t'), so every molecule silently came back
        # as None.
        with MolSupplier(self.smi_path) as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 2)
        smiles = [Chem.MolToSmiles(m) for _, m in mols]
        self.assertEqual(smiles, ['CCO', 'c1ccccc1'])

    def test_names_are_read_correctly(self):
        with open(self.smi_path) as fh:
            fh.readline()  # title line
            supplier = ForwardSmilesMolSupplier(fh, delimiter='\t', titleLine=False)
            mols = list(supplier)
        names = [m.GetProp('_Name') for m in mols]
        self.assertEqual(names, ['ethanol', 'benzene'])

    def test_no_name_column_does_not_raise(self):
        path = os.path.join(self._tmpdir.name, 'nonames.smi')
        with open(path, 'w') as f:
            f.write('SMILES\n')
            f.write('CCO\n')
            f.write('c1ccccc1\n')
        with MolSupplier(path) as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 2)


if __name__ == '__main__':
    unittest.main()
