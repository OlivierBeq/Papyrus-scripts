# -*- coding: utf-8 -*-

"""End-to-end integration tests for the object-oriented (PapyrusDataset) API.

Unlike the other files under tests/, these are NOT unit tests: they download
real Papyrus bioactivity datasets (one per version exercised below) and run
full filter -> aggregate pipelines against them, comparing the functional
reader/preprocess API against the equivalent PapyrusDataset chain. They
require network access and disk space, and are slow. Fast, offline unit
tests for the underlying reader/preprocess logic live in test_preprocess.py,
test_io.py, test_fingerprint.py and test_mol_reader.py instead.
"""

import unittest
from itertools import product

import numpy as np
import pandas as pd
from parameterized import parameterized, parameterized_class

from src.papyrus_scripts import PapyrusDataset
from src.papyrus_scripts import reader, preprocess


# Size of chunks of raw file to read
CHUNKSIZE = int(1e6)
# Path root directory of raw files (None = pystow's default)
SOURCE_PATH = None


def parametrized_test_name_func(testcase_func, _, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


def parametrized_testclass_name_func(cls, _, params_dict):
    return "{}_{}".format(
        cls.__name__,
        parameterized.to_safe_name("_".join(f'{k}_{v}' for k, v in params_dict.items())),
    )


@parameterized_class(
    ('stereo', 'version', 'plusplus'),
    list(product(
        [True, False],
        ['05.4', '05.5', '05.6'],
        [True, False]
    )), class_name_func=parametrized_testclass_name_func)
class TestPapyrusDataset(unittest.TestCase):
    """Each parametrization downloads (or reuses a cached download of) one
    Papyrus version/stereochemistry combination. Papyrus++ (plusplus=True)
    has no chiral (stereo=True) data, so that combination is expected to
    raise ValueError and is skipped after asserting so (see setUp).
    """

    def setUp(self):
        if self.plusplus and self.stereo:
            with self.assertRaises(ValueError):
                self._read_papyrus()
            self.skipTest('Papyrus++ has no stereochemistry (chiral) data')

    def _read_papyrus(self):
        return reader.read_papyrus(is3d=self.stereo, version=self.version, plusplus=self.plusplus,
                                   chunksize=CHUNKSIZE, source_path=SOURCE_PATH)

    def _read_protein_set(self):
        return reader.read_protein_set(version=self.version, source_path=SOURCE_PATH)

    def _new_dataset(self):
        return PapyrusDataset(is3d=self.stereo, version=self.version, plusplus=self.plusplus,
                              chunksize=CHUNKSIZE, source_path=SOURCE_PATH)

    def assertDataFrameEqual(self, df1: pd.DataFrame, df2: pd.DataFrame):
        # Ensure NaN values can be compared
        df1.fillna('NaN', inplace=True)
        df2.fillna('NaN', inplace=True)
        # Ensure dataframes are not empty
        self.assertFalse(df1.empty)
        self.assertFalse(df2.empty)
        # Check number of lines
        self.assertEqual(len(df1), len(df2))
        # Check number of columns
        self.assertEqual(df1.shape[1], df2.shape[1])
        # Check column names
        self.assertTrue((df1.columns == df2.columns).all())
        # Check content column by columns
        for j_col in range(df1.shape[1]):
            # First check dtype
            self.assertEqual(df1.iloc[:, j_col].dtype, df2.iloc[:, j_col].dtype)
            # Check content
            self.assertEqual(df1.iloc[:, j_col].tolist(),
                             df2.iloc[:, j_col].tolist())

    def test_medium_quality_kinase(self):
        # 1) Obtain data through the functional API
        fn_data = self._read_papyrus()
        fn_protein_data = self._read_protein_set()
        # Keep up to medium quality data (Papyrus++ only contains high quality)
        fn_filter1 = preprocess.keep_quality(fn_data, 'medium')
        # Keep kinases
        fn_filter2 = preprocess.keep_protein_class(fn_filter1, fn_protein_data,
                                                   classes={'l2': 'Kinase'})
        # Aggregate the data
        fn_data_agg = preprocess.consume_chunks(fn_filter2, progress=(not self.plusplus))
        # 2) Obtain data through the object-oriented API
        oop_data_agg = (self._new_dataset()
                        .keep_quality('medium')
                        .keep_protein_class({'l2': 'Kinase'})
                        .aggregate(progress=(not self.plusplus)))
        # 3) Ensure datasets are equal
        self.assertDataFrameEqual(fn_data_agg, oop_data_agg)
        del fn_protein_data
        # 4) Test values
        for quality in oop_data_agg.Quality.unique():
            self.assertIn(quality.lower(), ['high', 'medium'])
        self.assertEqual(oop_data_agg.Classification.str.split('->').str[1].unique(), ['Kinase'])

    def test_all_quality_human_adenosine_receptors_ic50(self):
        # 1) Obtain data through the functional API
        fn_data = self._read_papyrus()
        fn_protein_data = self._read_protein_set()
        # Keep human targets
        fn_filter1 = preprocess.keep_organism(fn_data, fn_protein_data,
                                              organism='Homo sapiens (Human)')
        # Keep adenosine receptors
        fn_filter2 = preprocess.keep_protein_class(fn_filter1, fn_protein_data,
                                                   classes={'l5': 'Adenosine receptor'})
        # Keep IC50
        fn_filter3 = preprocess.keep_type(fn_filter2, activity_types='ic50')
        # Aggregate the data
        fn_data_agg = preprocess.consume_chunks(fn_filter3, progress=(not self.plusplus))
        # 2) Obtain data through the object-oriented API
        oop_data_agg = (self._new_dataset()
                        .keep_organism('Homo sapiens (Human)')
                        .keep_protein_class({'l5': 'Adenosine receptor'})
                        .keep_activity_type('ic50')
                        .aggregate(progress=(not self.plusplus)))
        # 3) Ensure datasets are equal
        self.assertDataFrameEqual(fn_data_agg, oop_data_agg)
        del fn_data_agg
        # 4) Test values
        self.assertEqual(oop_data_agg.Classification.str.split('->').str[4].unique(), ['Adenosine receptor'])
        self.assertEqual(oop_data_agg.type_IC50.astype(int).unique().tolist(), [1])
        oop_data_proteins = (PapyrusDataset.from_dataframe(oop_data_agg, self.stereo, self.version, self.plusplus)
                             .proteins(progress=True)
                             .to_dataframe(False))
        self.assertEqual(len(oop_data_agg.accession.unique()), len(oop_data_proteins))
        self.assertEqual(oop_data_proteins.Organism.unique().tolist(), ['Homo sapiens (Human)'])

    def test_chembl_mouse_cc_chemokine_receptors_ki_and_kd(self):
        # 1) Obtain data through the functional API
        fn_data = self._read_papyrus()
        fn_protein_data = self._read_protein_set()
        # Keep ChEMBL data
        fn_filter1 = preprocess.keep_source(fn_data, 'chembl')
        # Keep human targets
        fn_filter2 = preprocess.keep_organism(fn_filter1, fn_protein_data,
                                              organism='Mus musculus (Mouse)')
        # Keep C-C chemokine receptors
        fn_filter3 = preprocess.keep_protein_class(fn_filter2, fn_protein_data,
                                                   classes={'l5': 'CC chemokine receptor'})
        # Drop CCL2 and CCL5
        fn_filter4 = preprocess.keep_not_match(fn_filter3, 'accession', ['P13500', 'P13501'])
        # Keep IC50
        fn_filter5 = preprocess.keep_type(fn_filter4, activity_types=['ki', 'kd'])
        # Aggregate the data
        fn_data_agg = preprocess.consume_chunks(fn_filter5, progress=(not self.plusplus))
        # 2) Obtain data through the object-oriented API
        oop_data_agg = (self._new_dataset()
                        .keep_source('chembl')
                        .keep_organism('Mus musculus (Mouse)')
                        .keep_protein_class({'l5': 'CC chemokine receptor'})
                        .not_isin('accession', ['P13500', 'P13501'])
                        .keep_activity_type(['ki', 'kd'])
                        .aggregate(progress=(not self.plusplus)))
        # 3) Ensure datasets are equal
        self.assertDataFrameEqual(fn_data_agg, oop_data_agg)
        del fn_data_agg
        # 4) Test values
        self.assertEqual(len(oop_data_agg.source.unique()), 1)
        self.assertTrue(oop_data_agg.source.unique().item().lower().startswith('chembl'))
        self.assertTrue(oop_data_agg.type_IC50.dropna().astype(int).unique().item() == 0)
        self.assertTrue(oop_data_agg.type_EC50.dropna().astype(int).unique().item() == 0)
        self.assertTrue(oop_data_agg.type_other.replace({'NA': np.nan, 'NaN': np.nan, 'nan': np.nan})
                        .dropna().empty or (oop_data_agg.type_other.replace({'NA': np.nan, 'NaN': np.nan, 'nan': np.nan})
                        .dropna().astype(int).unique().item() == 0))
        self.assertEqual((oop_data_agg[['type_KD', 'type_Ki']]
                          .astype(int).
                          drop_duplicates()
                          .apply(lambda x: sorted(x), axis=1)
                          .tolist()),
                         [[0, 1], [0, 1]]
                         )
        self.assertEqual(oop_data_agg.Classification.str.split('->').str[4].unique(), ['CC chemokine receptor'])
        for accession in oop_data_agg.accession.unique():
            self.assertNotIn(accession, ['P13500', 'P13501'])
        oop_data_proteins = (PapyrusDataset.from_dataframe(oop_data_agg, self.stereo, self.version, self.plusplus)
                             .proteins(progress=True)
                             .to_dataframe(False))
        self.assertEqual(oop_data_proteins.Organism.unique().tolist(), ['Mus musculus (Mouse)'])

    def test_sharma_klaeger_christman_egfr_specific_mutants_no_chirality(self):
        # 1) Obtain data through the functional API
        fn_data = self._read_papyrus()
        # Keep data related to the human EGFR from its accession
        fn_filter1 = preprocess.keep_accession(fn_data, 'P00533')
        # Keep specific mutants
        fn_filter2 = preprocess.keep_match(fn_filter1, 'target_id', ['P00533_L858R', 'P00533_L861Q'])
        # Keep only molecules without chiral centers
        fn_filter3 = preprocess.keep_contains(fn_filter2, 'InChIKey', 'UHFFFAOYSA')
        # Keep data from the Sharma, Klaeger and Christmann-Franck datasets
        fn_filter4 = preprocess.keep_source(fn_filter3, ['sharma', 'klaeger', 'christman'])
        # Keep only molecules without chiral centers
        fn_filter5 = preprocess.keep_not_contains(fn_filter4, 'InChIKey', '-O$', regex=True)
        # Aggregate the data
        fn_data_agg = preprocess.consume_chunks(fn_filter5, progress=(not self.plusplus))
        # 2) Obtain data through the object-oriented API
        oop_data_agg = (self._new_dataset()
                        .keep_accession('P00533')
                        .isin('target_id', ['P00533_L858R', 'P00533_L861Q'])
                        .contains('InChIKey', 'UHFFFAOYSA')
                        .keep_source(['sharma', 'klaeger', 'christman'])
                        .not_contains('InChIKey', '-O$', regex=True)
                        .aggregate(progress=(not self.plusplus)))
        # 3) Ensure datasets are equal
        self.assertDataFrameEqual(fn_data_agg, oop_data_agg)
        del fn_data_agg
        # 4) Test values
        self.assertEqual(oop_data_agg.accession.unique().item(), 'P00533')
        self.assertEqual(np.sort(oop_data_agg.target_id.unique()).tolist(), ['P00533_L858R', 'P00533_L861Q'])
        self.assertEqual(oop_data_agg.InChIKey.str.split('-').str[1].unique(), 'UHFFFAOYSA')
        self.assertNotEqual(oop_data_agg.InChIKey.str.split('-').str[2].unique(), 'O')


if __name__ == '__main__':
    unittest.main()
