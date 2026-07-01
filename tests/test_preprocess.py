# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.preprocess.

Unlike tests/test_oop.py, these tests do not download any Papyrus data:
they exercise the filtering/aggregation logic directly against small,
hand-built polars DataFrames that mimic the shape of the real dataset.

preprocess.py is polars-based end to end (no pandas, no chunked-iterator
dispatch per filter - only consume_chunks itself understands an iterator of
frames), so fixtures are pl.DataFrame/pl.Series and assertions use polars
idioms throughout.
"""

import unittest
from unittest.mock import MagicMock, patch

import polars as pl

from src.papyrus_scripts import preprocess as pp


def make_bioactivity_df():
    """A minimal bioactivity DataFrame with the columns keep_source/keep_type rely on."""
    return pl.DataFrame({
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
        self.df = pl.DataFrame({'Quality': ['High', 'Medium', 'Low']})

    def test_keep_high_only(self):
        result = pp.keep_quality(self.df, 'high')
        self.assertEqual(result['Quality'].to_list(), ['High'])

    def test_keep_medium_and_above(self):
        result = pp.keep_quality(self.df, 'medium')
        self.assertEqual(result['Quality'].to_list(), ['High', 'Medium'])

    def test_keep_low_and_above_keeps_all(self):
        result = pp.keep_quality(self.df, 'low')
        self.assertEqual(len(result), 3)

    def test_case_insensitive(self):
        result = pp.keep_quality(self.df, 'HIGH')
        self.assertEqual(result['Quality'].to_list(), ['High'])

    def test_invalid_quality_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_quality(self.df, 'bogus')

    def test_invalid_data_type_raises(self):
        # keep_quality does no input-type validation of its own (unlike the
        # `_supports_lazy`-decorated filters below) - a non-DataFrame input
        # fails naturally when `.filter` is called on it.
        with self.assertRaises(AttributeError):
            pp.keep_quality([1, 2, 3], 'high')


class TestKeepSource(unittest.TestCase):

    def setUp(self):
        self.df = make_bioactivity_df()

    def test_keep_single_source_splits_multi_source_rows(self):
        result = pp.keep_source(self.df, 'chembl')
        self.assertEqual(sorted(result['Activity_ID']), ['A1', 'A2'])
        # The multi-source row (A2) should be re-aggregated to only chembl's values.
        row = result.filter(pl.col('Activity_ID') == 'A2').to_dicts()[0]
        self.assertEqual(row['source'], 'chembl')
        # Aggregated values stay strings - no numeric coercion happens in the
        # polars pipeline (unlike the old pandas `pd.to_numeric` step).
        self.assertEqual(row['CID'], '2')
        self.assertEqual(row['pchembl_value'], '6.5')

    def test_keep_all_returns_input_untouched(self):
        result = pp.keep_source(self.df, 'all')
        self.assertTrue(result.equals(self.df))

    def test_source_not_in_data_returns_empty(self):
        result = pp.keep_source(self.df, 'unknown_source')
        self.assertTrue(result.is_empty())
        # Columns must be preserved even though the result is empty.
        self.assertListEqual(list(result.columns), list(self.df.columns))

    def test_invalid_data_type_raises(self):
        # keep_source is `_supports_lazy`-decorated: it validates its input
        # itself and raises TypeError (not ValueError) for a bad type.
        with self.assertRaises(TypeError):
            pp.keep_source([1, 2, 3], 'chembl')


class TestKeepType(unittest.TestCase):

    def setUp(self):
        self.df = make_bioactivity_df()

    def test_keep_single_type_splits_multi_type_rows(self):
        result = pp.keep_type(self.df, 'ic50')
        self.assertEqual(sorted(result['Activity_ID']), ['A1', 'A2'])
        row = result.filter(pl.col('Activity_ID') == 'A2').to_dicts()[0]
        # Aggregated values stay strings, same as keep_source.
        self.assertEqual(row['type_IC50'], '1')
        self.assertEqual(row['type_EC50'], '0')

    def test_keep_all_returns_input_untouched(self):
        result = pp.keep_type(self.df, 'all')
        self.assertTrue(result.equals(self.df))

    def test_keep_any_returns_input_untouched(self):
        result = pp.keep_type(self.df, 'any')
        self.assertTrue(result.equals(self.df))

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_type(self.df, 'bogus_type')

    def test_invalid_data_type_raises(self):
        with self.assertRaises(TypeError):
            pp.keep_type([1, 2, 3], 'ic50')

    def test_accepts_str_or_list(self):
        single = pp.keep_type(self.df, 'ic50')
        as_list = pp.keep_type(self.df, ['ic50'])
        self.assertEqual(sorted(single['Activity_ID']), sorted(as_list['Activity_ID']))


class TestKeepAccession(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({'target_id': ['P30542_WT', 'P00533_L858R', 'Q9Y5N1']})

    def test_single_accession(self):
        result = pp.keep_accession(self.df, 'P30542')
        self.assertEqual(result['target_id'].to_list(), ['P30542_WT'])

    def test_multiple_accessions(self):
        result = pp.keep_accession(self.df, ['P30542', 'P00533'])
        self.assertEqual(sorted(result['target_id']), ['P00533_L858R', 'P30542_WT'])

    def test_invalid_data_type_raises(self):
        with self.assertRaises(AttributeError):
            pp.keep_accession([1, 2, 3], 'P30542')


class TestKeepMatchAndNotMatch(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({'accession': ['P1', 'P2', 'P3']})

    def test_keep_match_single_value(self):
        result = pp.keep_match(self.df, 'accession', 'P1')
        self.assertEqual(result['accession'].to_list(), ['P1'])

    def test_keep_match_list_of_values(self):
        result = pp.keep_match(self.df, 'accession', ['P1', 'P2'])
        self.assertEqual(sorted(result['accession']), ['P1', 'P2'])

    def test_keep_not_match(self):
        result = pp.keep_not_match(self.df, 'accession', ['P1', 'P2'])
        self.assertEqual(result['accession'].to_list(), ['P3'])

    def test_keep_match_invalid_data_type_raises(self):
        with self.assertRaises(AttributeError):
            pp.keep_match([1, 2, 3], 'accession', 'P1')

    def test_keep_not_match_invalid_data_type_raises(self):
        with self.assertRaises(AttributeError):
            pp.keep_not_match([1, 2, 3], 'accession', 'P1')


class TestKeepContainsAndNotContains(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({'InChIKey': ['ABCDEF-UHFFFAOYSA-N', 'GHIJKL-UHFFFAOYSA-O', 'MNOPQR-XXXXX-N']})

    def test_keep_contains(self):
        result = pp.keep_contains(self.df, 'InChIKey', 'UHFFFAOYSA')
        self.assertEqual(len(result), 2)

    def test_keep_not_contains(self):
        result = pp.keep_not_contains(self.df, 'InChIKey', 'UHFFFAOYSA')
        self.assertEqual(result['InChIKey'].to_list(), ['MNOPQR-XXXXX-N'])

    def test_keep_contains_regex(self):
        result = pp.keep_contains(self.df, 'InChIKey', '-O$', regex=True)
        self.assertEqual(result['InChIKey'].to_list(), ['GHIJKL-UHFFFAOYSA-O'])

    def test_keep_contains_invalid_data_type_raises(self):
        with self.assertRaises(AttributeError):
            pp.keep_contains([1, 2, 3], 'InChIKey', 'UHFFFAOYSA')

    def test_keep_not_contains_invalid_data_type_raises(self):
        with self.assertRaises(AttributeError):
            pp.keep_not_contains([1, 2, 3], 'InChIKey', 'UHFFFAOYSA')


class TestKeepOrganism(unittest.TestCase):

    def setUp(self):
        self.data = pl.DataFrame({'target_id': ['P1', 'P2', 'P3']})
        self.protein_data = pl.DataFrame({
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
        self.assertTrue(result.equals(self.data))

    def test_generic_regex(self):
        result = pp.keep_organism(self.data, self.protein_data, organism='human', generic_regex=True)
        self.assertEqual(sorted(result['target_id']), ['P1', 'P3'])

    def test_invalid_data_type_raises(self):
        with self.assertRaises(TypeError):
            pp.keep_organism([1, 2, 3], self.protein_data)


class TestKeepProteinClass(unittest.TestCase):

    def setUp(self):
        self.data = pl.DataFrame({'target_id': ['P1', 'P2', 'P3']})
        self.protein_data = pl.DataFrame({
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
        self.assertEqual(result['target_id'].to_list(), ['P1'])

    def test_keep_by_deep_level(self):
        result = pp.keep_protein_class(self.data, self.protein_data, classes={'l5': 'Adenosine receptor'})
        self.assertEqual(result['target_id'].to_list(), ['P2'])

    def test_union_of_classes(self):
        result = pp.keep_protein_class(self.data, self.protein_data,
                                       classes=[{'l2': 'Kinase'}, {'l2': 'Protease'}])
        self.assertEqual(sorted(result['target_id']), ['P1', 'P3'])

    def test_level_independent_pattern(self):
        result = pp.keep_protein_class(self.data, self.protein_data, classes={'l?': 'kinase'})
        self.assertEqual(result['target_id'].to_list(), ['P1'])

    def test_none_classes_returns_all(self):
        result = pp.keep_protein_class(self.data, self.protein_data, classes=None)
        self.assertTrue(result.equals(self.data))

    def test_invalid_level_key_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_protein_class(self.data, self.protein_data, classes={'l9': 'Kinase'})

    def test_multiple_l_wildcard_keys_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_protein_class(self.data, self.protein_data, classes={'l?': 'Kinase', 'l1': 'Enzyme'})

    def test_invalid_data_type_raises(self):
        with self.assertRaises(TypeError):
            pp.keep_protein_class([1, 2, 3], self.protein_data)


class TestConsumeChunks(unittest.TestCase):

    def test_consume_flat_chunks(self):
        chunks = iter([pl.DataFrame({'a': [1, 2]}), pl.DataFrame({'a': [3, 4]})])
        result = pp.consume_chunks(chunks, progress=False)
        self.assertEqual(result['a'].to_list(), [1, 2, 3, 4])

    def test_consume_empty_iterator_returns_empty_dataframe(self):
        result = pp.consume_chunks(iter([]), progress=False)
        self.assertTrue(result.is_empty())

    def test_consume_nested_chunks(self):
        nested = iter([iter([pl.DataFrame({'a': [1]}), pl.DataFrame({'a': [2]})])])
        result = pp.consume_chunks(nested, progress=False)
        self.assertEqual(result['a'].to_list(), [1, 2])

    def test_consume_lazyframe_collects(self):
        lf = pl.DataFrame({'a': [1, 2, 3]}).lazy()
        result = pp.consume_chunks(lf, progress=False)
        self.assertEqual(result['a'].to_list(), [1, 2, 3])


class TestYScrambling(unittest.TestCase):

    def test_shuffles_single_column(self):
        data = pl.DataFrame({'pchembl_value_Mean': list(range(20))})
        original = data['pchembl_value_Mean'].to_list()
        result = pp.yscrambling(data, y_var='pchembl_value_Mean', random_state=42)
        self.assertNotEqual(result['pchembl_value_Mean'].to_list(), original)
        # Same multiset of values, just reordered.
        self.assertEqual(sorted(result['pchembl_value_Mean'].to_list()), sorted(original))

    def test_deterministic_given_seed(self):
        data = pl.DataFrame({'y': list(range(20))})
        result1 = pp.yscrambling(data.clone(), y_var='y', random_state=7)
        result2 = pp.yscrambling(data.clone(), y_var='y', random_state=7)
        self.assertEqual(result1['y'].to_list(), result2['y'].to_list())

    def test_multiple_columns(self):
        data = pl.DataFrame({'y1': list(range(10)), 'y2': list(range(10, 20))})
        result = pp.yscrambling(data, y_var=['y1', 'y2'], random_state=42)
        self.assertNotEqual(result['y1'].to_list(), list(range(10)))
        self.assertNotEqual(result['y2'].to_list(), list(range(10, 20)))

    def test_invalid_y_var_raises(self):
        data = pl.DataFrame({'y': [1, 2, 3]})
        with self.assertRaises(ValueError):
            pp.yscrambling(data, y_var=123)


class TestEqualizeCellSize(unittest.TestCase):
    """equalize_cell_size_in_row operates on a plain Python list (one value
    per column, addressed by position) rather than a pandas Series with a
    named index.
    """

    def test_equalize_row_internal(self):
        row = [[1, 2], [3], 'x']
        result = pp.equalize_cell_size_in_row(row, fill_mode='internal')
        self.assertEqual(result[0], [1, 2])
        self.assertEqual(result[1], [3, 3])
        self.assertEqual(result[2], ['x', 'x'])

    def test_equalize_row_external(self):
        row = [[1, 2], [3]]
        result = pp.equalize_cell_size_in_row(row, fill_mode='external', fill_value=0)
        self.assertEqual(result[0], [1, 2])
        self.assertEqual(result[1], [3, 0])

    def test_equalize_row_trim(self):
        row = [[1, 2, 3], [4, 5]]
        result = pp.equalize_cell_size_in_row(row, fill_mode='trim')
        self.assertEqual(result[0], [1, 2])
        self.assertEqual(result[1], [4, 5])

    def test_equalize_row_invalid_fill_mode_raises(self):
        with self.assertRaises(ValueError):
            pp.equalize_cell_size_in_row([[1, 2], [3]], fill_mode='bogus')

    def test_equalize_column_internal(self):
        col = pl.Series([[1, 2], [3]])
        result = pp.equalize_cell_size_in_column(col, fill_mode='internal')
        self.assertEqual(result.to_list(), [[1, 2], [3, 3]])

    def test_equalize_column_external(self):
        col = pl.Series([[1, 2], [3]])
        result = pp.equalize_cell_size_in_column(col, fill_mode='external', fill_value=0)
        self.assertEqual(result.to_list(), [[1, 2], [3, 0]])

    def test_equalize_column_trim(self):
        col = pl.Series([[1, 2, 3], [4, 5]])
        result = pp.equalize_cell_size_in_column(col, fill_mode='trim')
        self.assertEqual(result.to_list(), [[1, 2], [4, 5]])

    def test_equalize_column_invalid_fill_mode_raises(self):
        col = pl.Series([[1, 2], [3]])
        with self.assertRaises(ValueError):
            pp.equalize_cell_size_in_column(col, fill_mode='bogus')


class TestKeepSimilarDissimilarSubstructure(unittest.TestCase):
    """These tests mock FPSubSim2 entirely: building a real database is out of scope."""

    def test_keep_similar_missing_file_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_similar(pl.DataFrame(), 'CCO', '/no/such/file.h5')

    def test_keep_dissimilar_missing_file_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_dissimilar(pl.DataFrame(), 'CCO', '/no/such/file.h5')

    def test_keep_substructure_missing_file_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_substructure(pl.DataFrame(), 'CCO', '/no/such/file.h5')

    def test_keep_not_substructure_missing_file_raises(self):
        with self.assertRaises(ValueError):
            pp.keep_not_substructure(pl.DataFrame(), 'CCO', '/no/such/file.h5')

    @patch('src.papyrus_scripts.preprocess.Path.is_file', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_similar_unknown_fingerprint_raises(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        mock_fpss2.available_fingerprints = {'OtherFP': {}}
        mock_fpss2_cls.return_value = mock_fpss2
        with self.assertRaises(ValueError):
            pp.keep_similar(pl.DataFrame(), 'CCO', 'fake.h5')

    @patch('src.papyrus_scripts.preprocess.Path.is_file', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_similar_filters_by_similarity_result(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        mock_fpss2.available_fingerprints = {'Morgan_2048bits_0x0': {}}
        similar = pl.DataFrame({'InChIKey': ['KEY1', 'KEY2'], 'similarity': [0.9, 0.8]})
        mock_fpss2.get_similarity_lib.return_value.similarity.return_value = similar
        mock_fpss2_cls.return_value = mock_fpss2

        data = pl.DataFrame({'InChIKey': ['KEY1', 'KEY2', 'KEY3']})
        with patch('src.papyrus_scripts.preprocess.MorganFingerprint') as mock_fp_cls:
            # The lookup key is built from repr(fingerprint), not str(fingerprint).
            mock_fp_cls.return_value.__repr__ = lambda self: 'Morgan_2048bits_0x0'
            result = pp.keep_similar(data, 'CCO', 'fake.h5', fingerprint=mock_fp_cls.return_value)
        self.assertEqual(sorted(result['InChIKey']), ['KEY1', 'KEY2'])

    @patch('src.papyrus_scripts.preprocess.Path.is_file', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_dissimilar_filters_out_similarity_result(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        mock_fpss2.available_fingerprints = {'Morgan_2048bits_0x0': {}}
        similar = pl.DataFrame({'InChIKey': ['KEY1'], 'similarity': [0.9]})
        mock_fpss2.get_similarity_lib.return_value.similarity.return_value = similar
        mock_fpss2_cls.return_value = mock_fpss2

        data = pl.DataFrame({'InChIKey': ['KEY1', 'KEY2', 'KEY3']})
        with patch('src.papyrus_scripts.preprocess.MorganFingerprint') as mock_fp_cls:
            mock_fp_cls.return_value.__repr__ = lambda self: 'Morgan_2048bits_0x0'
            result = pp.keep_dissimilar(data, 'CCO', 'fake.h5', fingerprint=mock_fp_cls.return_value)
        self.assertEqual(sorted(result['InChIKey']), ['KEY2', 'KEY3'])

    @patch('src.papyrus_scripts.preprocess.Path.is_file', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_substructure_filters_by_substructure_result(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        substructures = pl.DataFrame({'InChIKey': ['KEY2']})
        mock_fpss2.get_substructure_lib.return_value.substructure.return_value = substructures
        mock_fpss2_cls.return_value = mock_fpss2

        data = pl.DataFrame({'InChIKey': ['KEY1', 'KEY2', 'KEY3']})
        result = pp.keep_substructure(data, 'CCO', 'fake.h5')
        self.assertEqual(result['InChIKey'].to_list(), ['KEY2'])

    @patch('src.papyrus_scripts.preprocess.Path.is_file', return_value=True)
    @patch('src.papyrus_scripts.preprocess.FPSubSim2')
    def test_keep_not_substructure_filters_out_substructure_result(self, mock_fpss2_cls, _mock_isfile):
        mock_fpss2 = MagicMock()
        substructures = pl.DataFrame({'InChIKey': ['KEY2']})
        mock_fpss2.get_substructure_lib.return_value.substructure.return_value = substructures
        mock_fpss2_cls.return_value = mock_fpss2

        data = pl.DataFrame({'InChIKey': ['KEY1', 'KEY2', 'KEY3']})
        result = pp.keep_not_substructure(data, 'CCO', 'fake.h5')
        self.assertEqual(sorted(result['InChIKey']), ['KEY1', 'KEY3'])


if __name__ == '__main__':
    unittest.main()