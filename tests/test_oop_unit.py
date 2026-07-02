# -*- coding: utf-8 -*-

"""Lightweight, non-network unit tests for papyrus_scripts.oop.

Unlike tests/test_oop.py (which downloads real Papyrus data end to end),
these tests build a PapyrusDataset directly from a small synthetic
pl.DataFrame via PapyrusDataset._from_data(), to exercise PapyrusDataFilter
without any network access.
"""

import unittest
from unittest.mock import patch

import polars as pl

from src.papyrus_scripts.oop import PapyrusDataset, PapyrusMoleculeSet, PapyrusProteinSet


def make_dataset():
    df = pl.DataFrame({
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
    protein_data = pl.DataFrame({
        'target_id': ['P1', 'P2'],
        'Organism': ['Homo sapiens (Human)', 'Homo sapiens (Human)'],
    })
    papyrus_params = dict(
        is3d=False, version=None, plusplus=True, chunksize=None,
        source_path=None, num_rows=len(df), download_progress=False,
    )
    return PapyrusDataset._from_data(
        papyrus_bioactivity_data=df,
        papyrus_protein_data=protein_data,
        papyrus_params=papyrus_params,
    )


class TestPapyrusDataFilterKeepSourceAndType(unittest.TestCase):
    """Regression test: PapyrusDataFilter.keep_source/keep_activity_type used to
    forward njobs=/verbose= kwargs to preprocess.keep_source/keep_type, which
    have never accepted them, raising TypeError on every call.
    """

    def setUp(self):
        self.dataset = make_dataset()

    def test_keep_source_does_not_raise(self):
        result = self.dataset.keep_source(source='chembl')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])

    def test_keep_activity_type_does_not_raise(self):
        result = self.dataset.keep_activity_type(activity_types='ic50')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])

    def test_keep_source_via_configured_filter_does_not_raise(self):
        # The class docstring's own documented usage pattern:
        # dataset._filter(njobs=4, progress=True).keep_quality('medium')
        result = self.dataset._filter(njobs=4, progress=True).keep_source(source='chembl')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])

    def test_keep_activity_type_via_configured_filter_does_not_raise(self):
        result = self.dataset._filter(njobs=2, progress=True).keep_activity_type(activity_types='ic50')
        ids = sorted(result.papyrus_bioactivity_data['Activity_ID'])
        self.assertEqual(ids, ['A1', 'A2'])


class TestPapyrusDatasetDownloadUsesConsistentFolderKey(unittest.TestCase):
    """Regression test: PapyrusDataset.__init__'s auto-download branch used
    to pass pv.version (the canonical new-format string, e.g. '2022.04.2')
    to download_papyrus, while every read afterwards (get_num_rows_in_file,
    read_papyrus, read_protein_set) used the original pv object directly -
    whose pystow_path_key is the *old*-format string (e.g. '05.4') when the
    caller supplied an old-format version. download_papyrus would then
    resolve its own PapyrusVersion from the canonical string and write to a
    different folder ('2022.04.2') than every read looked under ('05.4'),
    surfacing as a KeyError from a missing data_size.json after a
    multi-gigabyte download had already completed. Fixed by passing
    pv.pystow_path_key everywhere instead, so the write and every read agree.
    """

    def _run_init(self, version):
        with (
            patch('src.papyrus_scripts.oop.IO.is_local_version_available', return_value=False),
            patch('src.papyrus_scripts.oop.download.download_papyrus') as mock_download,
            patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', return_value=0),
            patch('src.papyrus_scripts.oop.reader.read_papyrus', return_value=pl.DataFrame()),
            patch('src.papyrus_scripts.oop.reader.read_protein_set', return_value=pl.DataFrame()),
        ):
            PapyrusDataset(version=version, download_progress=False)
        return mock_download

    def test_old_format_version_downloads_under_old_format_folder_key(self):
        mock_download = self._run_init('05.4')
        self.assertEqual(mock_download.call_args.kwargs['version'], '05.4')

    def test_new_format_version_downloads_under_new_format_folder_key(self):
        mock_download = self._run_init('2022.04.2')
        self.assertEqual(mock_download.call_args.kwargs['version'], '2022.04.2')


class TestDerivedSetReprShowsRealCount(unittest.TestCase):
    """Regression test: PapyrusMoleculeSet/PapyrusProteinSet.__repr__ checked
    isinstance(self.data, pd.DataFrame), but self.data is always a polars
    DataFrame (never pandas) for these two classes - the check never matched,
    so __repr__ always claimed "<iterator of X>" even for a concrete,
    already-materialised DataFrame.
    """

    def test_molecule_set_repr_shows_count_for_materialized_dataframe(self):
        df = pl.DataFrame({'connectivity': ['C1', 'C2'], 'mol': [None, None]})
        params = {'is3d': False, 'version': None, 'plusplus': True,
                  'chunksize': None, 'source_path': None}
        with patch('src.papyrus_scripts.oop.IO.get_num_rows_in_file', return_value=2):
            mset = PapyrusMoleculeSet(df, params)
        self.assertEqual(repr(mset), 'PapyrusMoleculeSet<2 molecules>')

    def test_protein_set_repr_shows_count_for_materialized_dataframe(self):
        df = pl.DataFrame({'target_id': ['P1', 'P2']})
        params = {'is3d': False, 'version': None, 'plusplus': True,
                  'chunksize': None, 'source_path': None}
        pset = PapyrusProteinSet(df, params, num_proteins=2)
        self.assertEqual(repr(pset), 'PapyrusProteinSet<2 proteins>')


if __name__ == '__main__':
    unittest.main()
