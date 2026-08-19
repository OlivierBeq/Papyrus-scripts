# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.utils.mol_reader.

All fixtures are small, real molecule files written to a temp directory
(no downloads, no mocking of RDKit): a couple of SMILES/SDF records are
enough to exercise the supplier/compression/format-detection logic.
"""

import bz2
import gzip
import io
import lzma
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem

from src.papyrus_scripts.utils import mol_reader as mr
from src.papyrus_scripts.utils.mol_reader import ForwardMol2MolSupplier, ForwardSmilesMolSupplier, MolSupplier

_MOL2_BLOCK = """@<TRIPOS>MOLECULE
ethanol
    3     2    0    0    0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1  LIG1        0.0000
      2 C2          1.5000    0.0000    0.0000 C.3     1  LIG1        0.0000
      3 O1          2.0000    1.0000    0.0000 O.3     1  LIG1        0.0000
@<TRIPOS>BOND
     1    1    2    1
     2    2    3    1
"""


class TestCompressionDetection(unittest.TestCase):
    """Compression/format detection moved from MolSupplier instance methods to
    module-level functions (plus one staticmethod) during the polars refactor.
    """

    def test_detects_lzma(self):
        label, open_fn, is_binary, inner_filename = mr._strip_compression_suffix('foo.sd.xz')
        self.assertEqual((label, inner_filename), ('lzma', 'foo.sd'))
        self.assertIs(open_fn, lzma.open)
        self.assertTrue(is_binary)

    def test_detects_zlib(self):
        label, open_fn, is_binary, inner_filename = mr._strip_compression_suffix('foo.sd.gz')
        self.assertEqual((label, inner_filename), ('zlib', 'foo.sd'))
        self.assertIs(open_fn, gzip.open)
        self.assertTrue(is_binary)

    def test_detects_bz2(self):
        label, open_fn, is_binary, inner_filename = mr._strip_compression_suffix('foo.sd.bz2')
        self.assertEqual((label, inner_filename), ('bz2', 'foo.sd'))
        self.assertIs(open_fn, bz2.open)
        self.assertTrue(is_binary)

    def test_no_compression(self):
        label, open_fn, is_binary, inner_filename = mr._strip_compression_suffix('foo.sd')
        self.assertEqual((label, inner_filename), (None, 'foo.sd'))
        self.assertIs(open_fn, open)
        self.assertFalse(is_binary)

    def test_compression_handler_mapping(self):
        self.assertEqual(MolSupplier._open_fn_for_label('lzma'), (lzma.open, True))
        self.assertEqual(MolSupplier._open_fn_for_label('zlib'), (gzip.open, True))
        self.assertEqual(MolSupplier._open_fn_for_label('bz2'), (bz2.open, True))

    def test_compression_handler_invalid_raises(self):
        with self.assertRaises(ValueError):
            MolSupplier._open_fn_for_label('bogus')


class TestFormatDetection(unittest.TestCase):

    def test_smi(self):
        self.assertEqual(mr._detect_format('foo.smi'), 'smi')

    def test_sd_and_sdf(self):
        self.assertEqual(mr._detect_format('foo.sd'), 'sd')
        self.assertEqual(mr._detect_format('foo.sdf'), 'sd')

    def test_mol2(self):
        self.assertEqual(mr._detect_format('foo.mol2'), 'mol2')

    def test_mol(self):
        self.assertEqual(mr._detect_format('foo.mol'), 'mol')

    def test_mae(self):
        self.assertEqual(mr._detect_format('foo.mae'), 'mae')

    def test_unknown_extension_raises(self):
        with self.assertRaises(ValueError):
            mr._detect_format('foo.bogus')


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
        # Unlike the other constructor validations (all ValueError), an
        # unrecognised `source` type falls through to a final TypeError.
        with self.assertRaises(TypeError):
            MolSupplier(source=123)


class SmallSDFFixture(unittest.TestCase):
    """Base class writing a small, real 2-molecule SDF fixture to a temp dir."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.sd_path = Path(self._tmpdir.name) / 'mols.sd'
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
        path = self.sd_path.with_name(self.sd_path.name + suffix)
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
        # auto-detected) used to crash with ValueError because the truncated
        # (suffix-stripped) filename was only ever derived on the auto-detect
        # path, so format detection ran against the still-compressed filename.
        path = self._compress(gzip.open, '.gz')
        with MolSupplier(path, compression='zlib') as ms:
            self.assertEqual(len(list(ms)), 2)


class TestMolSupplierSMI(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.smi_path = Path(self._tmpdir.name) / 'mols.smi'
        with open(self.smi_path, 'w') as f:
            f.write('SMILES\tName\n')
            f.write('CCO\tethanol\n')
            f.write('c1ccccc1\tbenzene\n')

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_reads_tab_delimited_smi_with_title_line(self):
        # Regression test: ForwardSmilesMolSupplier's manual text-streaming path
        # (the one MolSupplier always uses, since it always hands over an
        # already-open file handle) used to hardcode '\n' as RDKit's field
        # delimiter instead of the actual column delimiter (e.g. '\t'), so
        # every molecule silently came back as None - MolSupplier(path) would
        # yield zero molecules for any tab-delimited .smi file.
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
        path = Path(self._tmpdir.name) / 'nonames.smi'
        with open(path, 'w') as f:
            f.write('SMILES\n')
            f.write('CCO\n')
            f.write('c1ccccc1\n')
        with MolSupplier(path) as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 2)


class TestProcessedMolSupplierWarnings(unittest.TestCase):

    def test_multiple_unparseable_molecules_emit_a_single_summary_warning(self):
        # Regression test: one warnings.warn() per bad molecule used to spam
        # thousands of warnings on datasets with many unparseable entries.
        mol = Chem.MolFromSmiles('CCO')
        with MolSupplier(supplier=[mol, None, mol, None, None]) as ms:
            with self.assertWarns(UserWarning) as ctx:
                mols = list(ms)
        self.assertEqual(len(mols), 2)
        self.assertEqual(len(ctx.warnings), 1)
        self.assertIn('3 molecule(s)', str(ctx.warnings[0].message))

    def test_no_warning_when_all_molecules_parse(self):
        mol = Chem.MolFromSmiles('CCO')
        with MolSupplier(supplier=[mol, mol]) as ms:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                mols = list(ms)
        self.assertEqual(len(mols), 2)


class TestForwardMol2MolSupplier(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.mol2_path = Path(self._tmpdir.name) / 'mol.mol2'
        with open(self.mol2_path, 'w') as f:
            f.write(_MOL2_BLOCK)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_reads_single_record_from_filename(self):
        with ForwardMol2MolSupplier(self.mol2_path) as sup:
            mols = list(sup)
        self.assertEqual(len(mols), 1)
        self.assertIsNotNone(mols[0])

    def test_reads_from_open_text_handle(self):
        with open(self.mol2_path) as fh:
            sup = ForwardMol2MolSupplier(fh)
            mols = list(sup)
        self.assertEqual(len(mols), 1)

    def test_reads_multiple_records(self):
        path = Path(self._tmpdir.name) / 'multi.mol2'
        with open(path, 'w') as f:
            f.write(_MOL2_BLOCK + _MOL2_BLOCK)
        with ForwardMol2MolSupplier(path) as sup:
            mols = list(sup)
        self.assertEqual(len(mols), 2)

    def test_invalid_source_type_raises(self):
        with self.assertRaises(TypeError):
            ForwardMol2MolSupplier(123)

    def test_next_protocol(self):
        with ForwardMol2MolSupplier(self.mol2_path) as sup:
            mol = next(sup)
        self.assertIsNotNone(mol)

    def test_close_does_not_close_caller_owned_handle(self):
        with open(self.mol2_path) as fh:
            sup = ForwardMol2MolSupplier(fh)
            list(sup)
            sup.close()
            self.assertFalse(fh.closed)

    def test_reads_across_buffer_boundary(self):
        # A tiny buffer forces multiple read() calls before a separator is
        # found within the accumulated buffer.
        with patch.object(ForwardMol2MolSupplier, '_BUFFER_SIZE', 4):
            with ForwardMol2MolSupplier(self.mol2_path) as sup:
                mols = list(sup)
        self.assertEqual(len(mols), 1)


class TestForwardSmilesMolSupplierDirect(unittest.TestCase):
    """Direct construction, as opposed to via MolSupplier (which always
    hands over an already-open handle, never a filename).
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.smi_path = Path(self._tmpdir.name) / 'mols.smi'
        with open(self.smi_path, 'w') as f:
            f.write('SMILES\tName\n')
            f.write('CCO\tethanol\n')
            f.write('c1ccccc1\tbenzene\n')

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_constructed_from_filename(self):
        sup = ForwardSmilesMolSupplier(self.smi_path)
        mols = list(sup)
        self.assertEqual(len(mols), 2)

    def test_invalid_source_type_raises(self):
        with self.assertRaises(TypeError):
            ForwardSmilesMolSupplier(123)

    def test_next_protocol_from_handle(self):
        with open(self.smi_path) as fh:
            fh.readline()
            sup = ForwardSmilesMolSupplier(fh, delimiter='\t', titleLine=False)
            mol = next(sup)
        self.assertIsNotNone(mol)

    def test_close_is_a_no_op(self):
        sup = ForwardSmilesMolSupplier(self.smi_path)
        sup.close()  # must not raise

    def test_context_manager_from_handle(self):
        with open(self.smi_path) as fh:
            with ForwardSmilesMolSupplier(fh) as sup:
                mols = list(sup)
        self.assertEqual(len(mols), 2)

    def test_reads_across_buffer_boundary(self):
        with open(self.smi_path) as fh, patch.object(ForwardSmilesMolSupplier, '_BUFFER_SIZE', 4):
            sup = ForwardSmilesMolSupplier(fh)
            mols = list(sup)
        self.assertEqual(len(mols), 2)

    def test_last_line_without_trailing_newline_is_still_parsed(self):
        path = Path(self._tmpdir.name) / 'no_trailing_newline.smi'
        with open(path, 'w') as f:
            f.write('SMILES\tName\nCCO\tethanol')  # no final '\n'
        with open(path) as fh:
            mols = list(ForwardSmilesMolSupplier(fh))
        self.assertEqual(len(mols), 1)


class TestMolSupplierFileLikeSource(unittest.TestCase):

    def test_file_like_source_with_valid_format(self):
        handle = io.StringIO('SMILES\tName\nCCO\tethanol\n')
        with MolSupplier(handle, format='smi') as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 1)

    def test_file_like_source_with_invalid_format_raises(self):
        handle = io.StringIO('CCO')
        with self.assertRaises(ValueError):
            MolSupplier(handle, format='bogus')

    def test_mol2_format_dispatch(self):
        handle = io.StringIO(_MOL2_BLOCK)
        with MolSupplier(handle, format='mol2') as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 1)

    def test_mol2_binary_handle_raises_type_error(self):
        handle = io.BytesIO(_MOL2_BLOCK.encode())
        with self.assertRaises(TypeError):
            MolSupplier(handle, format='mol2')

    @patch('src.papyrus_scripts.utils.mol_reader.MaeMolSupplier')
    def test_mae_format_dispatches_to_mae_supplier(self, mock_cls):
        handle = io.BytesIO(b'dummy')
        MolSupplier(handle, format='mae')
        mock_cls.assert_called_once()


class TestMolSupplierIterationControl(unittest.TestCase):

    def test_set_start_progress_total(self):
        mol = Chem.MolFromSmiles('CCO')
        ms = MolSupplier(supplier=[mol])
        ms.set_start_progress_total(start=5, progress=False, total=1)
        self.assertEqual(ms._iter_start, 5)
        self.assertFalse(ms._iter_progress)
        self.assertEqual(ms._iter_total, 1)

    def test_show_progress_renders_tqdm_bar(self):
        mol = Chem.MolFromSmiles('CCO')
        with MolSupplier(supplier=[mol, mol], show_progress=True, total=2) as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 2)

    def test_next_protocol(self):
        mol = Chem.MolFromSmiles('CCO')
        with MolSupplier(supplier=[mol]) as ms:
            mol_id, rdmol = next(ms)
        self.assertEqual(mol_id, 0)
        self.assertIsNotNone(rdmol)


class TestMolSupplierParquet(SmallSDFFixture):
    """MolSupplier dispatches a '.parquet' source to ForwardParquetSDMolSupplier."""

    def setUp(self):
        super().setUp()
        from src.papyrus_scripts.utils.IO import convert_sd_to_parquet

        xz_path = self.sd_path.with_name(self.sd_path.name + '.xz')
        with open(self.sd_path, 'rb') as fin, lzma.open(xz_path, 'wb') as fout:
            fout.write(fin.read())
        self.parquet_path = Path(self._tmpdir.name) / 'mols.sd.parquet'
        convert_sd_to_parquet(xz_path, self.parquet_path)

    def test_reads_parquet_structures_file(self):
        with MolSupplier(self.parquet_path) as ms:
            mols = list(ms)
        self.assertEqual(len(mols), 2)
        smiles = sorted(Chem.MolToSmiles(m) for _, m in mols)
        self.assertEqual(smiles, sorted(['CCO', 'c1ccccc1']))

    def test_molecule_ids_start_at_zero_by_default(self):
        with MolSupplier(self.parquet_path) as ms:
            ids = [i for i, _ in ms]
        self.assertEqual(ids, [0, 1])

    def test_matches_reading_the_original_sdf(self):
        with MolSupplier(self.sd_path) as ms:
            from_sdf = [(i, Chem.MolToSmiles(m)) for i, m in ms]
        with MolSupplier(self.parquet_path) as ms:
            from_parquet = [(i, Chem.MolToSmiles(m)) for i, m in ms]
        self.assertEqual(from_sdf, from_parquet)

    def test_properties_round_trip(self):
        mol = Chem.MolFromSmiles('CN1CCC[C@H]1c1cccnc1')
        mol.SetProp('InChIKey', 'SNICXCGAKADSCV-JTQLQIEISA-N')
        writer = Chem.SDWriter(str(self.sd_path))
        writer.write(mol)
        writer.close()
        xz_path = self.sd_path.with_name('with_props.sd.xz')
        with open(self.sd_path, 'rb') as fin, lzma.open(xz_path, 'wb') as fout:
            fout.write(fin.read())
        parquet_path = Path(self._tmpdir.name) / 'with_props.sd.parquet'
        from src.papyrus_scripts.utils.IO import convert_sd_to_parquet
        convert_sd_to_parquet(xz_path, parquet_path)

        with MolSupplier(parquet_path) as ms:
            _, rdmol = next(iter(ms))
        self.assertEqual(rdmol.GetProp('InChIKey'), 'SNICXCGAKADSCV-JTQLQIEISA-N')


if __name__ == '__main__':
    unittest.main()
