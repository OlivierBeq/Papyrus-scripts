# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.matchRCSB.get_matches.

Mocks update_rcsb_data, polars.read_csv and pystow.module - no network or
real RCSB_data.tsv.xz file touched. Covers the data-type boundary (pandas/
polars DataFrame/LazyFrame input) and correctness of the match/aggregate
output (matching runs in polars, always returns pandas).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

from src.papyrus_scripts.matchRCSB import (
    _chunked_get_matches,
    get_all_pdb_ids_with_ligands,
    get_matches,
    papyrus_rcsb_data_root,
    update_rcsb_data,
)


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

    def test_inchikey_column_selects_inchi_3d_identifier(self):
        data = pl.DataFrame({
            'Activity_ID': ['A1'],
            'InChI': ['InChI=1_3D'],
            'accession': ['P1'],
            'InChIKey': ['KEY1'],
        })
        result = get_matches(data, update=True, verbose=False)
        self.assertEqual(result['PDBID_ligand'].tolist(), ['LIG1'])

    def test_accession_only_raises(self):
        data = pl.DataFrame({'Activity_ID': ['A1'], 'accession': ['P1']})
        with self.assertRaises(ValueError):
            get_matches(data, update=True, verbose=False)

    def test_no_identifying_column_raises(self):
        data = pl.DataFrame({'Activity_ID': ['A1']})
        with self.assertRaises(ValueError):
            get_matches(data, update=True, verbose=False)

    def test_rcsb_smiles_column_is_dropped_before_join(self):
        # rcsb_data carrying a SMILES column (as written by update_rcsb_data)
        # must not collide with the join - it's dropped first.
        rcsb_data = self.rcsb_data.with_columns(pl.lit('CCO').alias('SMILES'))
        with patch('polars.read_csv', return_value=rcsb_data):
            df = pl.DataFrame(make_bioactivity_data())
            result = get_matches(df, update=True, verbose=False)
        self.assertNotIn('SMILES', result.columns)


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


class TestPapyrusRcsbDataRoot(unittest.TestCase):

    def test_sets_root_folder_and_returns_rcsb_module(self):
        with (
            patch('src.papyrus_scripts.matchRCSB.IO._set_root_folder') as mock_set,
            patch('src.papyrus_scripts.matchRCSB.pystow.module') as mock_module,
        ):
            mock_module.return_value = 'rcsb-module'
            result = papyrus_rcsb_data_root('/some/root')
        mock_set.assert_called_once_with('/some/root')
        mock_module.assert_called_once_with('rcsb')
        self.assertEqual(result, 'rcsb-module')


class TestGetAllPdbIdsWithLigands(unittest.TestCase):

    def test_extracts_identifiers_from_result_set(self):
        session = MagicMock()
        session.post.return_value.json.return_value = {
            'result_set': [{'identifier': '1ABC'}, {'identifier': '2XYZ'}],
        }
        pdb_ids = get_all_pdb_ids_with_ligands(session)
        self.assertEqual(pdb_ids, ['1ABC', '2XYZ'])
        session.post.return_value.raise_for_status.assert_called_once()

    def test_no_session_creates_one(self):
        with patch('src.papyrus_scripts.matchRCSB.new_session') as mock_new_session:
            mock_new_session.return_value.post.return_value.json.return_value = {'result_set': []}
            pdb_ids = get_all_pdb_ids_with_ligands()
        mock_new_session.assert_called_once()
        self.assertEqual(pdb_ids, [])


def _graphql_entry(rcsb_id, ligand_id, smiles, desc_type='SMILES_STEREO', program='RDKit'):
    return {
        'rcsb_id': rcsb_id,
        'nonpolymer_entities': [{
            'nonpolymer_comp': {
                'chem_comp': {'id': ligand_id},
                'pdbx_chem_comp_descriptor': [
                    {'descriptor': smiles, 'type': desc_type, 'program': program},
                ],
            },
        }],
    }


class TestUpdateRcsbData(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_path = Path(self._tmpdir.name) / 'RCSB_data.tsv.xz'
        self.root_patch = patch('src.papyrus_scripts.matchRCSB.papyrus_rcsb_data_root')
        mock_root = self.root_patch.start()
        mock_root.return_value.join.return_value = self.output_path
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self._tmpdir.cleanup)
        self.sleep_patch = patch('src.papyrus_scripts.matchRCSB.time.sleep')
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_recent_file_is_reused_without_fetching(self):
        pd.DataFrame({'PDBID_ligand': ['LIG1']}).to_csv(self.output_path, sep='\t', index=False)
        with patch('src.papyrus_scripts.matchRCSB.new_session') as mock_new_session:
            result = update_rcsb_data(verbose=False)
        mock_new_session.assert_not_called()
        self.assertEqual(result['PDBID_ligand'].tolist(), ['LIG1'])

    def test_recent_file_prints_message_when_verbose(self):
        pd.DataFrame({'PDBID_ligand': ['LIG1']}).to_csv(self.output_path, sep='\t', index=False)
        update_rcsb_data(verbose=True)  # should not raise

    def _mock_session(self, ligand_ids, graphql_status=200):
        session = MagicMock()
        session.post.return_value.status_code = graphql_status
        session.post.return_value.json.return_value = {
            'data': {'entries': [_graphql_entry('PDB1', lid, 'CCO') for lid in ligand_ids]},
        }
        return session

    def _run(self, ligand_ids, verbose, graphql_status=200, uniprot_map=None):
        session = self._mock_session(ligand_ids, graphql_status)
        uniprot_map = uniprot_map if uniprot_map is not None else pd.DataFrame(
            {'PDB': ['PDB1'], 'UniProtKB_AC-ID': ['P1']},
        )
        with (
            patch('src.papyrus_scripts.matchRCSB.new_session', return_value=session),
            patch(
                'src.papyrus_scripts.matchRCSB.get_all_pdb_ids_with_ligands',
                return_value=['PDB1'],
            ),
            patch(
                'src.papyrus_scripts.matchRCSB.UniprotMatch.uniprot_mappings',
                return_value=uniprot_map,
            ),
        ):
            return update_rcsb_data(overwrite=True, verbose=verbose)

    def test_fetches_and_writes_matches(self):
        result = self._run(['LIG1'], verbose=False)
        self.assertEqual(result['PDBID_ligand'].tolist(), ['LIG1'])
        self.assertEqual(result['UniProt_accession'].tolist(), ['P1'])
        self.assertTrue(self.output_path.is_file())

    def test_verbose_progress_bar_path(self):
        result = self._run(['LIG1'], verbose=True)
        self.assertEqual(len(result), 1)

    def test_non_200_response_is_skipped(self):
        result = self._run(['LIG1'], verbose=False, graphql_status=500)
        self.assertTrue(result.empty)

    def test_non_200_response_verbose_writes_to_progress_bar(self):
        result = self._run(['LIG1'], verbose=True, graphql_status=500)
        self.assertTrue(result.empty)

    def test_entry_without_nonpolymer_entities_is_skipped(self):
        session = MagicMock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {
            'data': {'entries': [{'rcsb_id': 'PDB1', 'nonpolymer_entities': None}]},
        }
        with (
            patch('src.papyrus_scripts.matchRCSB.new_session', return_value=session),
            patch(
                'src.papyrus_scripts.matchRCSB.get_all_pdb_ids_with_ligands',
                return_value=['PDB1'],
            ),
        ):
            result = update_rcsb_data(overwrite=True, verbose=False)
        self.assertTrue(result.empty)

    def test_entry_without_nonpolymer_comp_is_skipped(self):
        session = MagicMock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {
            'data': {'entries': [{
                'rcsb_id': 'PDB1',
                'nonpolymer_entities': [{'nonpolymer_comp': None}],
            }]},
        }
        with (
            patch('src.papyrus_scripts.matchRCSB.new_session', return_value=session),
            patch(
                'src.papyrus_scripts.matchRCSB.get_all_pdb_ids_with_ligands',
                return_value=['PDB1'],
            ),
        ):
            result = update_rcsb_data(overwrite=True, verbose=False)
        self.assertTrue(result.empty)

    def test_entry_without_descriptors_is_skipped(self):
        session = MagicMock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {
            'data': {'entries': [{
                'rcsb_id': 'PDB1',
                'nonpolymer_entities': [{
                    'nonpolymer_comp': {'chem_comp': {'id': 'LIG1'}, 'pdbx_chem_comp_descriptor': None},
                }],
            }]},
        }
        with (
            patch('src.papyrus_scripts.matchRCSB.new_session', return_value=session),
            patch(
                'src.papyrus_scripts.matchRCSB.get_all_pdb_ids_with_ligands',
                return_value=['PDB1'],
            ),
        ):
            result = update_rcsb_data(overwrite=True, verbose=False)
        self.assertTrue(result.empty)

    def test_openeye_stereo_descriptor_short_circuits_search(self):
        # Two descriptors: a non-OpenEye one first, then the OpenEye
        # SMILES_STEREO one that must win and stop the inner loop (break).
        session = MagicMock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {
            'data': {'entries': [{
                'rcsb_id': 'PDB1',
                'nonpolymer_entities': [{
                    'nonpolymer_comp': {
                        'chem_comp': {'id': 'LIG1'},
                        'pdbx_chem_comp_descriptor': [
                            {'descriptor': 'CCO', 'type': 'SMILES_CANONICAL', 'program': 'Other'},
                            {'descriptor': 'CCO', 'type': 'SMILES_STEREO', 'program': 'OpenEye OEToolkits'},
                        ],
                    },
                }],
            }]},
        }
        with (
            patch('src.papyrus_scripts.matchRCSB.new_session', return_value=session),
            patch(
                'src.papyrus_scripts.matchRCSB.get_all_pdb_ids_with_ligands',
                return_value=['PDB1'],
            ),
            patch(
                'src.papyrus_scripts.matchRCSB.UniprotMatch.uniprot_mappings',
                return_value=pd.DataFrame({'PDB': ['PDB1'], 'UniProtKB_AC-ID': ['P1']}),
            ),
        ):
            result = update_rcsb_data(overwrite=True, verbose=False)
        self.assertEqual(result['PDBID_ligand'].tolist(), ['LIG1'])

    def test_many_pdb_ids_sleeps_between_requests(self):
        # > chunk_size (200) pdb_ids so the rate-limit sleep after each
        # chunk is actually exercised.
        pdb_ids = [f'PDB{i}' for i in range(201)]
        session = self._mock_session(['LIG1'])
        with (
            patch('src.papyrus_scripts.matchRCSB.time.sleep') as mock_sleep,
            patch('src.papyrus_scripts.matchRCSB.new_session', return_value=session),
            patch(
                'src.papyrus_scripts.matchRCSB.get_all_pdb_ids_with_ligands',
                return_value=pdb_ids,
            ),
            patch(
                'src.papyrus_scripts.matchRCSB.UniprotMatch.uniprot_mappings',
                return_value=pd.DataFrame({'PDB': ['PDB1'], 'UniProtKB_AC-ID': ['P1']}),
            ),
        ):
            update_rcsb_data(overwrite=True, verbose=False)
        mock_sleep.assert_called()


class TestChunkedGetMatches(unittest.TestCase):

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

    def test_get_matches_dispatches_iterator_input_to_chunked_helper(self):
        chunks = iter([pd.DataFrame(make_bioactivity_data())])
        result = get_matches(chunks, verbose=False)
        processed = list(result)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]['PDBID_ligand'].tolist(), ['LIG1'])

    def test_chunked_get_matches_yields_one_result_per_chunk(self):
        chunks = [pd.DataFrame(make_bioactivity_data()), pd.DataFrame(make_bioactivity_data())]
        results = list(_chunked_get_matches(chunks, root_folder=None, verbose=False, total=None))
        self.assertEqual(len(results), 2)

    def test_chunked_get_matches_verbose_uses_progress_bar(self):
        chunks = [pd.DataFrame(make_bioactivity_data())]
        results = list(_chunked_get_matches(chunks, root_folder=None, verbose=True, total=1))
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()
