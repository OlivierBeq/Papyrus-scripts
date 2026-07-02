# -*- coding: utf-8 -*-

"""Extensive offline tests for papyrus_scripts.reader.

Builds a small, realistic Papyrus version folder entirely locally (no
network access, no real Papyrus download) and exercises every reader.py
entry point against it: bioactivities (2D/3D/Papyrus++), protein targets,
every molecular descriptor type (mold2, CDDD, mordred 2D/3D, ECFP6/E3FP,
MOE, 'all'), protein descriptors (unirep, custom), and molecular structures
(SD files, 2D/3D, chunked reading).

The fixture is built twice per scenario: once left as the raw ``.tsv.xz``
layout download_papyrus writes before conversion (exercising reader.py's
`_open_source`/`.xz` fallback path), and once with every tabular file
converted to ``.parquet`` and the ``.xz`` deleted, mirroring
download_papyrus's default behavior (exercising the Parquet-preferred
path). Both must produce identical results.

``data_types.json`` is built using the *real* plain lowercase type-name
string format Papyrus actually ships (e.g. ``"float"``, ``"int"``), not the
TypeEncoder/__type__ hydrated form - using the real format is what
originally surfaced the to_polars_dtype bug this suite guards against
(see TestDataTypeNameStringSchema in test_io.py).
"""

import json
import lzma
import tempfile
import unittest
import warnings
from pathlib import Path

import polars as pl
from rdkit import Chem

from src.papyrus_scripts import reader
from src.papyrus_scripts.utils.IO import convert_xz_to_parquet

#: A real alias+revision from the bundled aliases.json (offline-resolvable),
#: reused only as a folder key inside an isolated tmp pystow home - never
#: touches the real ~/.data directory.
VERSION = '2022.04.2'
#: Numeric-dot-numeric filename prefix matching reader.py's `\d+\.\d+...`
#: patterns; the actual digits are irrelevant to those patterns.
PREFIX = '05.4'

# 2D bioactivity molecules/targets shared across all 2D descriptor fixtures
# so a join (desc_type='all') has overlapping rows to return.
_CONNECTIVITY_IDS = ['CONN1', 'CONN2', 'CONN3']
_INCHIKEY_IDS = ['INCHI1', 'INCHI2']
_TARGET_IDS = ['P1', 'P2']


def _write_xz_tsv(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, 'wb') as fh:
        fh.write(df.write_csv(separator='\t').encode())


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(data, fh)


def _write_sd_xz(path: Path, mols: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plain_path = path.with_suffix('')
    writer = Chem.SDWriter(str(plain_path))
    for mol in mols:
        writer.write(mol)
    writer.close()
    with open(plain_path, 'rb') as fh:
        raw = fh.read()
    plain_path.unlink()
    with lzma.open(path, 'wb') as fh:
        fh.write(raw)


def build_fixture(root: Path) -> None:
    """Populate *root* as a pystow home containing one full Papyrus version.

    :param root: empty directory to use as PYSTOW_HOME-equivalent
        (``source_path``/``root_folder`` argument)
    """
    version_dir = root / 'papyrus' / VERSION
    desc_dir = version_dir / 'descriptors'
    struct_dir = version_dir / 'structures'

    (root / 'papyrus').mkdir(parents=True, exist_ok=True)
    with open(root / 'papyrus' / 'versions.json', 'w') as fh:
        json.dump([VERSION], fh)

    # ------------------------------------------------------------------
    # Bioactivities: full 2D, full 3D, and a *different*, smaller Papyrus++
    # subset - deliberately distinct row sets so a test can prove
    # read_papyrus(plusplus=...) actually picks the right file.
    # ------------------------------------------------------------------
    bioactivity_2d = pl.DataFrame({
        'Activity_ID': [f'A{i}' for i in range(1, 6)],
        'connectivity': (_CONNECTIVITY_IDS + _CONNECTIVITY_IDS)[:5],
        'target_id': (_TARGET_IDS * 3)[:5],
        'Quality': ['High', 'Medium', 'Low', 'High', 'Medium'],
        'source': ['chembl'] * 5,
        'pchembl_value_Mean': [6.5, 7.1, 5.0, 8.2, 6.9],
    })
    _write_xz_tsv(
        version_dir / f'{PREFIX}_combined_set_without_stereochemistry.tsv.xz',
        bioactivity_2d,
    )

    bioactivity_pp = pl.DataFrame({
        'Activity_ID': ['P1', 'P2', 'P3'],
        'connectivity': _CONNECTIVITY_IDS,
        'target_id': _TARGET_IDS + [_TARGET_IDS[0]],
        'Quality': ['High', 'High', 'High'],
        'source': ['chembl'] * 3,
        'pchembl_value_Mean': [7.0, 7.5, 8.0],
    })
    _write_xz_tsv(
        version_dir / f'{PREFIX}++_combined_set_without_stereochemistry.tsv.xz',
        bioactivity_pp,
    )

    bioactivity_3d = pl.DataFrame({
        'Activity_ID': ['A1_3D', 'A2_3D', 'A3_3D'],
        'InChIKey': _INCHIKEY_IDS + [_INCHIKEY_IDS[0]],
        'target_id': _TARGET_IDS + [_TARGET_IDS[0]],
        'Quality': ['High', 'Medium', 'High'],
        'source': ['chembl'] * 3,
        'pchembl_value_Mean': [6.0, 6.6, 7.7],
    })
    _write_xz_tsv(
        version_dir / f'{PREFIX}_combined_set_with_stereochemistry.tsv.xz',
        bioactivity_3d,
    )

    # ------------------------------------------------------------------
    # Protein targets
    # ------------------------------------------------------------------
    protein_data = pl.DataFrame({
        'target_id': _TARGET_IDS,
        'UniProtID': ['P1_HUMAN', 'P2_HUMAN'],
        'Organism': ['Homo sapiens (Human)', 'Homo sapiens (Human)'],
        'Sequence': ['MKV', 'MAG'],
    })
    _write_xz_tsv(
        version_dir / f'{PREFIX}_combined_set_protein_targets.tsv.xz',
        protein_data,
    )

    # ------------------------------------------------------------------
    # Molecular descriptors - 2D (keyed by connectivity, same 3 ids
    # everywhere so desc_type='all' can inner-join non-empty results)
    # ------------------------------------------------------------------
    mold2 = pl.DataFrame({
        'connectivity': _CONNECTIVITY_IDS,
        'D001': [1, 2, 3],
        'D002': [4, 5, 6],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_2D_moldescs_mold2.tsv.xz', mold2)

    cddd = pl.DataFrame({
        'connectivity': _CONNECTIVITY_IDS,
        'CDDD_1': [0.1, 0.2, 0.3],
        'CDDD_2': [0.4, 0.5, 0.6],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_2D_moldescs_CDDDs.tsv.xz', cddd)

    mordred_2d = pl.DataFrame({
        'connectivity': _CONNECTIVITY_IDS,
        'ABC': [1.1, 2.2, 3.3],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_2D_moldescs_mordred2D.tsv.xz', mordred_2d)

    ecfp6 = pl.DataFrame({
        'connectivity': _CONNECTIVITY_IDS,
        'ECFP6_1': [0, 1, 1],
        'ECFP6_2': [1, 0, 1],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_2D_moldescs_ECFP6.tsv.xz', ecfp6)

    moe_2d = pl.DataFrame({
        'connectivity': _CONNECTIVITY_IDS,
        'MOE_1': [9.9, 8.8, 7.7],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_2D_moldescs_MOE.tsv.xz', moe_2d)

    # ------------------------------------------------------------------
    # Molecular descriptors - 3D (keyed by InChIKey)
    # ------------------------------------------------------------------
    mordred_3d = pl.DataFrame({
        'InChIKey': _INCHIKEY_IDS,
        'ABC': [4.4, 5.5],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_3D_moldescs_mordred3D.tsv.xz', mordred_3d)

    e3fp = pl.DataFrame({
        'InChIKey': _INCHIKEY_IDS,
        'E3FP_1': [1, 0],
        'E3FP_2': [0, 1],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_3D_moldescs_E3FP.tsv.xz', e3fp)

    moe_3d = pl.DataFrame({
        'InChIKey': _INCHIKEY_IDS,
        'MOE_1': [6.6, 5.5],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_3D_moldescs_MOE.tsv.xz', moe_3d)

    # ------------------------------------------------------------------
    # Protein descriptors - unirep
    # ------------------------------------------------------------------
    unirep = pl.DataFrame({
        'TARGET_NAME': _TARGET_IDS,
        'UniRep64_AH_1': [0.11, 0.22],
        'UniRep64_AH_2': [0.33, 0.44],
    })
    _write_xz_tsv(desc_dir / f'{PREFIX}_combined_prot_embeddings_unirep.tsv.xz', unirep)

    # ------------------------------------------------------------------
    # Molecular structures - 2D and 3D SD files
    # ------------------------------------------------------------------
    mols_2d = []
    for smi, conn in zip(['CCO', 'c1ccccc1', 'CCN'], _CONNECTIVITY_IDS):
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp('connectivity', conn)
        mols_2d.append(mol)
    _write_sd_xz(
        struct_dir / f'{PREFIX}_combined_2D_set_without_stereochemistry.sd.xz', mols_2d,
    )

    mols_3d = []
    for smi, inchikey in zip(['CCO', 'c1ccccc1'], _INCHIKEY_IDS):
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp('InChIKey', inchikey)
        mols_3d.append(mol)
    _write_sd_xz(
        struct_dir / f'{PREFIX}_combined_3D_set_with_stereochemistry.sd.xz', mols_3d,
    )

    # ------------------------------------------------------------------
    # data_types.json - real shipped format: plain lowercase type names,
    # not TypeEncoder's __type__-hydrated form.
    # ------------------------------------------------------------------
    _write_json(version_dir / 'data_types.json', {
        'papyrus': {
            'Activity_ID': 'str', 'connectivity': 'str', 'InChIKey': 'str',
            'target_id': 'str', 'Quality': 'str', 'source': 'str',
            'pchembl_value_Mean': 'float',
        },
        'mold2': {'connectivity': 'str', 'D001': 'int', 'D002': 'int'},
        'CDDD': {'connectivity': 'str', 'CDDD_1': 'float', 'CDDD_2': 'float'},
        'mordred_2D': {'connectivity': 'str', 'ABC': 'float'},
        'mordred_3D': {'InChIKey': 'str', 'ABC': 'float'},
        'ECFP6': {'connectivity': 'str', 'ECFP6_1': 'int', 'ECFP6_2': 'int'},
        'E3FP': {'InChIKey': 'str', 'E3FP_1': 'int', 'E3FP_2': 'int'},
        'unirep': {
            'TARGET_NAME': 'str', 'UniRep64_AH_1': 'float', 'UniRep64_AH_2': 'float',
        },
    })

    # ------------------------------------------------------------------
    # data_size.json
    # ------------------------------------------------------------------
    _write_json(version_dir / 'data_size.json', {
        'papyrus_++': 3, 'papyrus_2D': 5, 'papyrus_3D': 3, 'papyrus_proteins': 2,
        'structures_2D': 3, 'structures_3D': 2,
        'mold2': 3, 'CDDD': 3, 'mordred_2D': 3, 'mordred_3D': 2,
        'ECFP6': 3, 'E3FP': 2, 'unirep': 2,
    })


def convert_all_tabular_to_parquet(root: Path) -> None:
    """Convert every ``.tsv.xz`` under *root* to ``.parquet`` and delete the
    original, mirroring download_papyrus's default (``keep_xz=False``)
    behavior. Structure files (``.sd.xz``) are left untouched.
    """
    version_dir = root / 'papyrus' / VERSION
    for xz_path in list(version_dir.rglob('*.tsv.xz')):
        parquet_path = xz_path.with_suffix('.parquet')
        convert_xz_to_parquet(xz_path, parquet_path, separator='\t')
        xz_path.unlink()


# ---------------------------------------------------------------------------
# Shared test bodies, run once against the raw .xz layout and once against
# the Parquet-converted layout (see the two concrete TestCase subclasses
# below).
# ---------------------------------------------------------------------------

class _ReaderOfflineTests:
    """Mixin with every test_* method; ROOT is set by concrete subclasses."""

    ROOT: str

    # -- bioactivities -----------------------------------------------

    def test_read_papyrus_2d_full(self):
        df = reader.read_papyrus(is3d=False, plusplus=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['Activity_ID']), ['A1', 'A2', 'A3', 'A4', 'A5'])

    def test_read_papyrus_2d_plusplus_picks_the_right_file(self):
        df = reader.read_papyrus(is3d=False, plusplus=True, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['Activity_ID']), ['P1', 'P2', 'P3'])

    def test_read_papyrus_3d(self):
        df = reader.read_papyrus(is3d=True, plusplus=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['Activity_ID']), ['A1_3D', 'A2_3D', 'A3_3D'])

    def test_read_papyrus_3d_plusplus_raises(self):
        with self.assertRaises(ValueError):
            reader.read_papyrus(is3d=True, plusplus=True, version=VERSION, source_path=self.ROOT)

    def test_read_papyrus_lazy_matches_eager(self):
        eager = reader.read_papyrus(is3d=False, plusplus=False, version=VERSION, source_path=self.ROOT)
        lazy = reader.read_papyrus(
            is3d=False, plusplus=False, version=VERSION, source_path=self.ROOT, chunksize=1,
        )
        self.assertIsInstance(lazy, pl.LazyFrame)
        self.assertTrue(lazy.collect().equals(eager))

    def test_read_papyrus_schema_override_applied(self):
        # Regression guard: pchembl_value_Mean must come back as Float64 -
        # not Utf8 - proving data_types.json's plain string type names
        # ("float", not a TypeEncoder __type__ marker) are honored.
        df = reader.read_papyrus(is3d=False, plusplus=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(df.schema['pchembl_value_Mean'], pl.Float64)

    # -- protein targets -----------------------------------------------

    def test_read_protein_set(self):
        df = reader.read_protein_set(version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['target_id']), _TARGET_IDS)
        self.assertEqual(set(df.columns), {'target_id', 'UniProtID', 'Organism', 'Sequence'})

    # -- molecular descriptors: individual types ------------------------

    def test_molecular_descriptors_mold2(self):
        df = reader.read_molecular_descriptors(desc_type='mold2', is3d=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['connectivity']), _CONNECTIVITY_IDS)
        self.assertEqual(df.schema['D001'], pl.Int64)

    def test_molecular_descriptors_cddd(self):
        df = reader.read_molecular_descriptors(desc_type='cddd', is3d=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(df.schema['CDDD_1'], pl.Float64)

    def test_molecular_descriptors_mordred_2d(self):
        df = reader.read_molecular_descriptors(desc_type='mordred', is3d=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['connectivity']), _CONNECTIVITY_IDS)

    def test_molecular_descriptors_mordred_3d(self):
        df = reader.read_molecular_descriptors(desc_type='mordred', is3d=True, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['InChIKey']), _INCHIKEY_IDS)

    def test_molecular_descriptors_fingerprint_2d_ecfp6(self):
        df = reader.read_molecular_descriptors(desc_type='fingerprint', is3d=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(df.schema['ECFP6_1'], pl.Int64)

    def test_molecular_descriptors_fingerprint_3d_e3fp(self):
        df = reader.read_molecular_descriptors(desc_type='fingerprint', is3d=True, version=VERSION, source_path=self.ROOT)
        self.assertEqual(df.schema['E3FP_1'], pl.Int64)

    def test_molecular_descriptors_moe_2d(self):
        df = reader.read_molecular_descriptors(desc_type='moe', is3d=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['connectivity']), _CONNECTIVITY_IDS)

    def test_molecular_descriptors_moe_3d(self):
        df = reader.read_molecular_descriptors(desc_type='moe', is3d=True, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['InChIKey']), _INCHIKEY_IDS)

    def test_molecular_descriptors_ids_filter(self):
        df = reader.read_molecular_descriptors(
            desc_type='mold2', is3d=False, version=VERSION, source_path=self.ROOT,
            ids=['CONN1'],
        )
        self.assertEqual(df['connectivity'].to_list(), ['CONN1'])

    def test_molecular_descriptors_lazy_matches_eager(self):
        eager = reader.read_molecular_descriptors(desc_type='mold2', is3d=False, version=VERSION, source_path=self.ROOT)
        lazy = reader.read_molecular_descriptors(
            desc_type='mold2', is3d=False, version=VERSION, source_path=self.ROOT, chunksize=1,
        )
        self.assertIsInstance(lazy, pl.LazyFrame)
        self.assertTrue(lazy.collect().equals(eager))

    # -- molecular descriptors: 'all' (join across every desc type) -----

    def test_molecular_descriptors_all_2d(self):
        df = reader.read_molecular_descriptors(desc_type='all', is3d=False, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['connectivity']), _CONNECTIVITY_IDS)
        for col in ('D001', 'CDDD_1', 'ABC', 'ECFP6_1', 'MOE_1'):
            self.assertIn(col, df.columns)

    def test_molecular_descriptors_all_3d(self):
        df = reader.read_molecular_descriptors(desc_type='all', is3d=True, version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['InChIKey']), _INCHIKEY_IDS)
        for col in ('ABC', 'E3FP_1', 'MOE_1'):
            self.assertIn(col, df.columns)

    def test_molecular_descriptors_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            reader.read_molecular_descriptors(desc_type='not_a_real_type', version=VERSION, source_path=self.ROOT)

    # -- protein descriptors: unirep + custom ---------------------------

    def test_protein_descriptors_unirep(self):
        df = reader.read_protein_descriptors(desc_type='unirep', version=VERSION, source_path=self.ROOT)
        self.assertEqual(sorted(df['target_id']), _TARGET_IDS)
        self.assertNotIn('TARGET_NAME', df.columns)
        self.assertEqual(df.schema['UniRep64_AH_1'], pl.Float64)

    def test_protein_descriptors_unirep_ids_filter(self):
        df = reader.read_protein_descriptors(
            desc_type='unirep', version=VERSION, source_path=self.ROOT, ids=['P1'],
        )
        self.assertEqual(df['target_id'].to_list(), ['P1'])

    def test_protein_descriptors_custom(self):
        with tempfile.TemporaryDirectory() as d:
            custom_path = Path(d) / 'custom.tsv'
            pl.DataFrame({
                'TARGET_NAME': _TARGET_IDS, 'feature': [1, 2],
            }).write_csv(custom_path, separator='\t')
            df = reader.read_protein_descriptors(desc_type='custom', source_path=str(custom_path))
        self.assertIn('target_id', df.columns)
        self.assertEqual(sorted(df['target_id']), _TARGET_IDS)

    def test_protein_descriptors_custom_missing_file_raises(self):
        with self.assertRaises(ValueError):
            reader.read_protein_descriptors(desc_type='custom', source_path='/nonexistent/path.tsv')

    def test_protein_descriptors_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            reader.read_protein_descriptors(desc_type='not_a_real_type', version=VERSION, source_path=self.ROOT)

    # -- molecular structures --------------------------------------------

    def test_molecular_structures_2d(self):
        df = reader.read_molecular_structures(is3d=False, version=VERSION, source_path=self.ROOT, verbose=False)
        self.assertEqual(len(df), 3)
        self.assertEqual(sorted(df['connectivity']), _CONNECTIVITY_IDS)

    def test_molecular_structures_3d(self):
        df = reader.read_molecular_structures(is3d=True, version=VERSION, source_path=self.ROOT, verbose=False)
        self.assertEqual(len(df), 2)
        self.assertEqual(sorted(df['InChIKey']), _INCHIKEY_IDS)

    def test_molecular_structures_ids_filter(self):
        df = reader.read_molecular_structures(
            is3d=False, version=VERSION, source_path=self.ROOT, verbose=False, ids=['CONN1'],
        )
        self.assertEqual(df['connectivity'].to_list(), ['CONN1'])

    def test_molecular_structures_chunked(self):
        chunks = list(reader.read_molecular_structures(
            is3d=False, version=VERSION, source_path=self.ROOT, chunksize=2, verbose=False,
        ))
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 3)
        self.assertTrue(all(len(c) <= 2 for c in chunks))


def _make_root() -> tempfile.TemporaryDirectory:
    tmpdir = tempfile.TemporaryDirectory()
    build_fixture(Path(tmpdir.name))
    return tmpdir


class TestReaderAgainstRawXzFiles(_ReaderOfflineTests, unittest.TestCase):
    """Every test in _ReaderOfflineTests against the raw .tsv.xz/.sd.xz
    layout download_papyrus writes before conversion - exercises reader.py's
    `_open_source` LZMA-decompression fallback path.
    """

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        cls._tmpdir = _make_root()
        cls.ROOT = cls._tmpdir.name

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()


class TestReaderAgainstParquetFiles(_ReaderOfflineTests, unittest.TestCase):
    """Every test in _ReaderOfflineTests against the layout after every
    tabular file has been converted to .parquet and the .xz deleted (the
    default download_papyrus behavior) - exercises the Parquet-preferred
    `_scan_tabular`/`_prefer_parquet` path. Structure files stay .sd.xz
    (Parquet conversion never applies to them).
    """

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        cls._tmpdir = _make_root()
        cls.ROOT = cls._tmpdir.name
        convert_all_tabular_to_parquet(Path(cls.ROOT))

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()


if __name__ == '__main__':
    unittest.main()
