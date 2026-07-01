# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.preprocess.

Unlike tests/test_oop.py, these tests do not download any Papyrus data:
they exercise the filtering/aggregation logic directly against small,
hand-built DataFrames that mimic the shape of the real dataset.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.papyrus_scripts import preprocess as pp


def make_bioactivity_df():
    """A minimal bioactivity DataFrame with the columns keep_source/keep_type rely on."""
    return pd.DataFrame({
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


class TestKeepQuality(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'Quality': ['High', 'Medium', 'Low']})

    def test_keep_high_only(self):
        result = pp.keep_quality(self.df, 'high')
        self.assertEqual(result['Quality'].tolist(), ['High'])

    def test_keep_medium_and_above(self):
        result = pp.keep_quality(self.df, 'medium')
        self.assertEqual(result['Quality'].tolist(), ['High', 'Medium'])

    def test_keep_low_and_above_keeps_all(self):
        result = pp.keep_quality(self.df, 'low')
        self.assertEqual(len(result), 3)

    def test_case_insensitive(self):
        result = pp.keep_quality(self.df, 'HIGH')
        self.assertEqual(result['Quality'].tolist(), ['High'])

    def test_invalid_quality_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_quality(self.df, 'bogus')

    def test_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_quality([1, 2, 3], 'high')

    def test_chunked_input(self):
        chunks = iter([self.df.iloc[:2], self.df.iloc[2:]])
        result = list(pp.keep_quality(chunks, 'medium'))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['Quality'].tolist(), ['High', 'Medium'])
        self.assertTrue(result[1].empty)


class TestKeepSource(unittest.TestCase):

    def setUp(self):
        self.df = make_bioactivity_df()

    def test_keep_single_source_splits_multi_source_rows(self):
        result = pp.keep_source(self.df, 'chembl', njobs=1, verbose=False)
        self.assertEqual(sorted(result['Activity_ID']), ['A1', 'A2'])
        # The multi-source row (A2) should be re-aggregated to only chembl's values.
        row = result[result['Activity_ID'] == 'A2'].iloc[0]
        self.assertEqual(row['source'], 'chembl')
        # The aggregation path runs values through pd.to_numeric, so numeric-looking
        # strings come back as numeric dtypes rather than the original strings.
        self.assertEqual(row['CID'], 2)
        self.assertEqual(row['pchembl_value'], 6.5)

    def test_keep_all_returns_input_untouched(self):
        result = pp.keep_source(self.df, 'all', njobs=1, verbose=False)
        pd.testing.assert_frame_equal(result, self.df)

    def test_source_not_in_data_returns_empty(self):
        result = pp.keep_source(self.df, 'unknown_source', njobs=1, verbose=False)
        self.assertTrue(result.empty)
        # Columns must be preserved even though the result is empty.
        self.assertListEqual(list(result.columns), list(self.df.columns))

    def test_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_source([1, 2, 3], 'chembl')

    def test_chunked_input(self):
        chunks = iter([self.df.iloc[:2].copy(), self.df.iloc[2:].copy()])
        result = list(pp.keep_source(chunks, 'chembl', njobs=1, verbose=False))
        self.assertEqual(len(result), 2)


class TestKeepType(unittest.TestCase):

    def setUp(self):
        self.df = make_bioactivity_df()

    def test_keep_single_type_splits_multi_type_rows(self):
        result = pp.keep_type(self.df, 'ic50', njobs=1, verbose=False)
        self.assertEqual(sorted(result['Activity_ID']), ['A1', 'A2'])
        row = result[result['Activity_ID'] == 'A2'].iloc[0]
        # The aggregation path runs values through pd.to_numeric, so numeric-looking
        # strings come back as numeric dtypes rather than the original strings.
        self.assertEqual(row['type_IC50'], 1)
        self.assertEqual(row['type_EC50'], 0)

    def test_keep_all_returns_input_untouched(self):
        result = pp.keep_type(self.df, 'all', njobs=1, verbose=False)
        pd.testing.assert_frame_equal(result, self.df)

    def test_keep_any_returns_input_untouched(self):
        result = pp.keep_type(self.df, 'any', njobs=1, verbose=False)
        pd.testing.assert_frame_equal(result, self.df)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_type(self.df, 'bogus_type')

    def test_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_type([1, 2, 3], 'ic50')

    def test_accepts_str_or_list(self):
        single = pp.keep_type(self.df, 'ic50', njobs=1, verbose=False)
        as_list = pp.keep_type(self.df, ['ic50'], njobs=1, verbose=False)
        self.assertEqual(sorted(single['Activity_ID']), sorted(as_list['Activity_ID']))


class TestKeepAccession(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'target_id': ['P30542_WT', 'P00533_L858R', 'Q9Y5N1']})

    def test_single_accession(self):
        result = pp.keep_accession(self.df, 'P30542')
        self.assertEqual(result['target_id'].tolist(), ['P30542_WT'])

    def test_multiple_accessions(self):
        result = pp.keep_accession(self.df, ['P30542', 'P00533'])
        self.assertEqual(sorted(result['target_id']), ['P00533_L858R', 'P30542_WT'])

    def test_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_accession([1, 2, 3], 'P30542')

    def test_chunked_input(self):
        chunks = iter([self.df.iloc[:2], self.df.iloc[2:]])
        result = list(pp.keep_accession(chunks, 'P30542'))
        self.assertEqual(len(result), 2)


class TestKeepMatchAndNotMatch(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'accession': ['P1', 'P2', 'P3']})

    def test_keep_match_single_value(self):
        result = pp.keep_match(self.df, 'accession', 'P1')
        self.assertEqual(result['accession'].tolist(), ['P1'])

    def test_keep_match_list_of_values(self):
        result = pp.keep_match(self.df, 'accession', ['P1', 'P2'])
        self.assertEqual(sorted(result['accession']), ['P1', 'P2'])

    def test_keep_not_match(self):
        result = pp.keep_not_match(self.df, 'accession', ['P1', 'P2'])
        self.assertEqual(result['accession'].tolist(), ['P3'])

    def test_keep_match_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_match([1, 2, 3], 'accession', 'P1')

    def test_keep_not_match_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_not_match([1, 2, 3], 'accession', 'P1')

    def test_keep_match_chunked_input(self):
        chunks = iter([self.df.iloc[:2], self.df.iloc[2:]])
        result = list(pp.keep_match(chunks, 'accession', 'P1'))
        self.assertEqual(len(result), 2)

    def test_keep_not_match_chunked_input(self):
        chunks = iter([self.df.iloc[:2], self.df.iloc[2:]])
        result = list(pp.keep_not_match(chunks, 'accession', 'P1'))
        self.assertEqual(len(result), 2)


class TestKeepContainsAndNotContains(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'InChIKey': ['ABCDEF-UHFFFAOYSA-N', 'GHIJKL-UHFFFAOYSA-O', 'MNOPQR-XXXXX-N']})

    def test_keep_contains(self):
        result = pp.keep_contains(self.df, 'InChIKey', 'UHFFFAOYSA')
        self.assertEqual(len(result), 2)

    def test_keep_not_contains(self):
        result = pp.keep_not_contains(self.df, 'InChIKey', 'UHFFFAOYSA')
        self.assertEqual(result['InChIKey'].tolist(), ['MNOPQR-XXXXX-N'])

    def test_keep_contains_regex(self):
        result = pp.keep_contains(self.df, 'InChIKey', '-O$', regex=True)
        self.assertEqual(result['InChIKey'].tolist(), ['GHIJKL-UHFFFAOYSA-O'])

    def test_keep_contains_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_contains([1, 2, 3], 'InChIKey', 'UHFFFAOYSA')

    def test_keep_not_contains_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_not_contains([1, 2, 3], 'InChIKey', 'UHFFFAOYSA')

    def test_keep_contains_chunked_input(self):
        chunks = iter([self.df.iloc[:2], self.df.iloc[2:]])
        result = list(pp.keep_contains(chunks, 'InChIKey', 'UHFFFAOYSA'))
        self.assertEqual(len(result), 2)


class TestKeepOrganism(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({'target_id': ['P1', 'P2', 'P3']})
        self.protein_data = pd.DataFrame({
            'target_id': ['P1', 'P2', 'P3'],
            'Organism': ['Homo sapiens (Human)', 'Mus musculus (Mouse)', 'Homo sapiens (Human)'],
        })

    def test_keep_single_organism(self):
        result = pp.keep_organism(self.data, self.protein_data, organism='Homo sapiens (Human)')
        self.assertEqual(sorted(result['target_id']), ['P1', 'P3'])
        self.assertTrue((result['Organism'] == 'Homo sapiens (Human)').all())

    def test_keep_multiple_organisms(self):
        result = pp.keep_organism(self.data, self.protein_data,
                                  organism=['Homo sapiens (Human)', 'Mus musculus (Mouse)'])
        self.assertEqual(len(result), 3)

    def test_none_organism_returns_all(self):
        result = pp.keep_organism(self.data, self.protein_data, organism=None)
        pd.testing.assert_frame_equal(result, self.data)

    def test_generic_regex(self):
        result = pp.keep_organism(self.data, self.protein_data, organism='human', generic_regex=True)
        self.assertEqual(sorted(result['target_id']), ['P1', 'P3'])

    def test_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_organism([1, 2, 3], self.protein_data)

    def test_chunked_input(self):
        chunks = iter([self.data.iloc[:2], self.data.iloc[2:]])
        result = list(pp.keep_organism(chunks, self.protein_data, organism='Homo sapiens (Human)'))
        self.assertEqual(len(result), 2)


class TestKeepProteinClass(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({'target_id': ['P1', 'P2', 'P3']})
        self.protein_data = pd.DataFrame({
            'target_id': ['P1', 'P2', 'P3'],
            'Classification': [
                'Enzyme->Kinase',
                'Membrane receptor->Family A G protein-coupled receptor->Small molecule receptor'
                '->Nucleotide-like receptor->Adenosine receptor',
                'Enzyme->Protease',
            ],
        })

    def test_keep_by_level(self):
        result = pp.keep_protein_class(self.data, self.protein_data, classes={'l2': 'Kinase'})
        self.assertEqual(result['target_id'].tolist(), ['P1'])

    def test_keep_by_deep_level(self):
        result = pp.keep_protein_class(self.data, self.protein_data, classes={'l5': 'Adenosine receptor'})
        self.assertEqual(result['target_id'].tolist(), ['P2'])

    def test_union_of_classes(self):
        result = pp.keep_protein_class(self.data, self.protein_data,
                                       classes=[{'l2': 'Kinase'}, {'l2': 'Protease'}])
        self.assertEqual(sorted(result['target_id']), ['P1', 'P3'])

    def test_level_independent_pattern(self):
        result = pp.keep_protein_class(self.data, self.protein_data, classes={'l?': 'kinase'})
        self.assertEqual(result['target_id'].tolist(), ['P1'])

    def test_none_classes_returns_all(self):
        result = pp.keep_protein_class(self.data, self.protein_data, classes=None)
        pd.testing.assert_frame_equal(result, self.data)

    def test_invalid_level_key_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_protein_class(self.data, self.protein_data, classes={'l9': 'Kinase'})

    def test_multiple_l_wildcard_keys_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_protein_class(self.data, self.protein_data, classes={'l?': 'Kinase', 'l1': 'Enzyme'})

    def test_invalid_data_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_protein_class([1, 2, 3], self.protein_data)


class TestConsumeChunks(unittest.TestCase):

    def test_consume_flat_chunks(self):
        chunks = iter([pd.DataFrame({'a': [1, 2]}), pd.DataFrame({'a': [3, 4]})])
        result = pp.consume_chunks(chunks, progress=False)
        self.assertEqual(result['a'].tolist(), [1, 2, 3, 4])

    def test_consume_empty_iterator_returns_empty_dataframe(self):
        result = pp.consume_chunks(iter([]), progress=False)
        self.assertTrue(result.empty)

    def test_consume_nested_chunks(self):
        nested = iter([iter([pd.DataFrame({'a': [1]}), pd.DataFrame({'a': [2]})])])
        result = pp.consume_chunks(nested, progress=False)
        self.assertEqual(result['a'].tolist(), [1, 2])


class TestYScrambling(unittest.TestCase):

    def test_shuffles_single_column(self):
        data = pd.DataFrame({'pchembl_value_Mean': list(range(20))})
        original = data['pchembl_value_Mean'].tolist()
        result = pp.yscrambling(data, y_var='pchembl_value_Mean', random_state=42)
        self.assertNotEqual(result['pchembl_value_Mean'].tolist(), original)
        # Same multiset of values, just reordered.
        self.assertEqual(sorted(result['pchembl_value_Mean'].tolist()), sorted(original))

    def test_deterministic_given_seed(self):
        data = pd.DataFrame({'y': list(range(20))})
        result1 = pp.yscrambling(data.copy(), y_var='y', random_state=7)
        result2 = pp.yscrambling(data.copy(), y_var='y', random_state=7)
        self.assertEqual(result1['y'].tolist(), result2['y'].tolist())

    def test_shuffles_when_index_is_not_default(self):
        # Regression test: shuffle() preserves index labels, so assigning the
        # shuffled Series back must not be silently undone by index alignment.
        data = pd.DataFrame({'y': list(range(20))}, index=list(range(100, 120)))
        original = data['y'].tolist()
        result = pp.yscrambling(data, y_var='y', random_state=42)
        self.assertNotEqual(result['y'].tolist(), original)

    def test_multiple_columns(self):
        data = pd.DataFrame({'y1': list(range(10)), 'y2': list(range(10, 20))})
        result = pp.yscrambling(data, y_var=['y1', 'y2'], random_state=42)
        self.assertNotEqual(result['y1'].tolist(), list(range(10)))
        self.assertNotEqual(result['y2'].tolist(), list(range(10, 20)))

    def test_invalid_y_var_raises(self):
        data = pd.DataFrame({'y': [1, 2, 3]})
        with self.assertRaises(ValueError):
            pp.yscrambling(data, y_var=123)


class TestEqualizeCellSize(unittest.TestCase):

    def test_equalize_row_internal(self):
        row = pd.Series([[1, 2], [3], 'x'], index=['a', 'b', 'c'])
        result = pp.equalize_cell_size_in_row(row, fill_mode='internal')
        self.assertEqual(result['a'], [1, 2])
        self.assertEqual(result['b'], [3, 3])
        self.assertEqual(result['c'], ['x', 'x'])

    def test_equalize_row_external(self):
        row = pd.Series([[1, 2], [3]], index=['a', 'b'])
        result = pp.equalize_cell_size_in_row(row, fill_mode='external', fill_value=0)
        self.assertEqual(result['a'], [1, 2])
        self.assertEqual(result['b'], [3, 0])

    def test_equalize_row_trim(self):
        row = pd.Series([[1, 2, 3], [4, 5]], index=['a', 'b'])
        result = pp.equalize_cell_size_in_row(row, fill_mode='trim')
        self.assertEqual(result['a'], [1, 2])
        self.assertEqual(result['b'], [4, 5])

    def test_equalize_row_invalid_fill_mode_raises(self):
        row = pd.Series([[1, 2], [3]], index=['a', 'b'])
        with self.assertRaises(ValueError):
            pp.equalize_cell_size_in_row(row, fill_mode='bogus')

    def test_equalize_column_internal(self):
        col = pd.Series([[1, 2], [3]])
        result = pp.equalize_cell_size_in_column(col, fill_mode='internal')
        self.assertEqual(result.tolist(), [[1, 2], [3, 3]])

    def test_equalize_column_external(self):
        col = pd.Series([[1, 2], [3]])
        result = pp.equalize_cell_size_in_column(col, fill_mode='external', fill_value=0)
        self.assertEqual(result.tolist(), [[1, 2], [3, 0]])

    def test_equalize_column_trim(self):
        col = pd.Series([[1, 2, 3], [4, 5]])
        result = pp.equalize_cell_size_in_column(col, fill_mode='trim')
        self.assertEqual(result.tolist(), [[1, 2], [4, 5]])

    def test_equalize_column_invalid_fill_mode_raises(self):
        col = pd.Series([[1, 2], [3]])
        with self.assertRaises(ValueError):
            pp.equalize_cell_size_in_column(col, fill_mode='bogus')


class TestKeepSimilarDissimilarSubstructure(unittest.TestCase):
    """These tests mock FPSubSim2 entirely: building a real database is out of scope."""

    def test_keep_similar_missing_file_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_similar(pd.DataFrame(), 'CCO', '/no/such/file.h5')

    def test_keep_dissimilar_missing_file_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_dissimilar(pd.DataFrame(), 'CCO', '/no/such/file.h5')

    def test_keep_substructure_missing_file_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_substructure(pd.DataFrame(), 'CCO', '/no/such/file.h5')

    def test_keep_not_substructure_missing_file_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_not_substructure(pd.DataFrame(), 'CCO', '/no/such/file.h5')

    @patch('src.papyrus_scripts.preprocess.os.path.isfile', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_similar_unknown_fingerprint_raises(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        mock_fpss2.available_fingerprints = {'OtherFP': {}}
        mock_fpss2_cls.return_value = mock_fpss2
        with self.assertRaises(ValueError):
            pp.keep_similar(pd.DataFrame(), 'CCO', 'fake.h5')

    @patch('src.papyrus_scripts.preprocess.os.path.isfile', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_similar_filters_by_similarity_result(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        mock_fpss2.available_fingerprints = {'Morgan_2048bits_0x0': {}}
        similar = pd.DataFrame({'other_col': [1, 2], 'similarity': [0.9, 0.8],
                                'InChIKey': ['KEY1', 'KEY2']})
        mock_fpss2.get_similarity_lib.return_value.similarity.return_value = similar
        mock_fpss2_cls.return_value = mock_fpss2

        data = pd.DataFrame({'InChIKey': ['KEY1', 'KEY2', 'KEY3']})
        with patch('src.papyrus_scripts.preprocess.MorganFingerprint') as mock_fp_cls:
            mock_fp_cls.return_value.__str__.return_value = 'Morgan_2048bits_0x0'
            result = pp.keep_similar(data, 'CCO', 'fake.h5', fingerprint=mock_fp_cls.return_value)
        self.assertEqual(sorted(result['InChIKey']), ['KEY1', 'KEY2'])

    @patch('src.papyrus_scripts.preprocess.os.path.isfile', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_dissimilar_filters_out_similarity_result(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        mock_fpss2.available_fingerprints = {'Morgan_2048bits_0x0': {}}
        similar = pd.DataFrame({'other_col': [1], 'similarity': [0.9], 'InChIKey': ['KEY1']})
        mock_fpss2.get_similarity_lib.return_value.similarity.return_value = similar
        mock_fpss2_cls.return_value = mock_fpss2

        data = pd.DataFrame({'InChIKey': ['KEY1', 'KEY2', 'KEY3']})
        with patch('src.papyrus_scripts.preprocess.MorganFingerprint') as mock_fp_cls:
            mock_fp_cls.return_value.__str__.return_value = 'Morgan_2048bits_0x0'
            result = pp.keep_dissimilar(data, 'CCO', 'fake.h5', fingerprint=mock_fp_cls.return_value)
        self.assertEqual(sorted(result['InChIKey']), ['KEY2', 'KEY3'])

    @patch('src.papyrus_scripts.preprocess.os.path.isfile', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_substructure_filters_by_substructure_result(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        substructures = pd.DataFrame({'InChIKey': ['KEY2']})
        mock_fpss2.get_substructure_lib.return_value.substructure.return_value = substructures
        mock_fpss2_cls.return_value = mock_fpss2

        data = pd.DataFrame({'InChIKey': ['KEY1', 'KEY2', 'KEY3']})
        result = pp.keep_substructure(data, 'CCO', 'fake.h5')
        self.assertEqual(result['InChIKey'].tolist(), ['KEY2'])

    @patch('src.papyrus_scripts.preprocess.os.path.isfile', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_not_substructure_filters_out_substructure_result(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        substructures = pd.DataFrame({'InChIKey': ['KEY2']})
        mock_fpss2.get_substructure_lib.return_value.substructure.return_value = substructures
        mock_fpss2_cls.return_value = mock_fpss2

        data = pd.DataFrame({'InChIKey': ['KEY1', 'KEY2', 'KEY3']})
        result = pp.keep_not_substructure(data, 'CCO', 'fake.h5')
        self.assertEqual(sorted(result['InChIKey']), ['KEY1', 'KEY3'])


if __name__ == '__main__':
    unittest.main()
