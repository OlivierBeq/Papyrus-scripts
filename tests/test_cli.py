# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.cli, the click-based command line
interface. Underlying library calls (download_papyrus, FPSubSim2,
process_data_version, ...) are mocked throughout - these tests exercise
argument parsing/dispatch, not the functionality behind it.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.papyrus_scripts.cli import main


class TestDownloadCommandDiskMargin(unittest.TestCase):

    def _invoke(self, *args):
        runner = CliRunner()
        with patch('src.papyrus_scripts.cli.download_papyrus') as mock_download:
            result = runner.invoke(main, ['download', *args])
        self.assertEqual(result.exit_code, 0, result.output)
        return mock_download

    def test_defaults_to_ten_percent(self):
        mock_download = self._invoke()
        self.assertEqual(mock_download.call_args.kwargs['disk_margin'], 0.1)

    def test_force_disables_the_margin(self):
        mock_download = self._invoke('--force')
        self.assertEqual(mock_download.call_args.kwargs['disk_margin'], 0.0)


class TestFpsubsim2CommandVersionedOutput(unittest.TestCase):
    """Each --version value must get its own --output path; a multi-version
    run must not overwrite one version's file with the next.
    """

    def _invoke(self, *args):
        runner = CliRunner()
        with patch('src.papyrus_scripts.cli.FPSubSim2') as mock_cls:
            result = runner.invoke(main, ['fpsubsim2', *args])
        self.assertEqual(result.exit_code, 0, result.output)
        return mock_cls.return_value.create_from_papyrus

    def test_multi_version_derives_distinct_output_files(self):
        mock_create = self._invoke(
            '--output', 'foo.h5', '--version', '2022.04.2', '--version', '2024.09.2',
            '--fingerprint', 'none',
        )
        outfiles = [call.kwargs['outfile'] for call in mock_create.call_args_list]
        self.assertEqual(outfiles, ['foo_2022.04.2.h5', 'foo_2024.09.2.h5'])

    def test_single_version_keeps_exact_output(self):
        mock_create = self._invoke(
            '--output', 'foo.h5', '--version', '2022.04.2', '--fingerprint', 'none',
        )
        self.assertEqual(mock_create.call_args.kwargs['outfile'], 'foo.h5')


class TestCleanCommand(unittest.TestCase):

    def test_forwards_options_to_remove_papyrus(self):
        runner = CliRunner()
        with patch('src.papyrus_scripts.cli.remove_papyrus') as mock_remove:
            result = runner.invoke(main, [
                'clean', '--version', '2022.04.2', '--version', '2024.09.2',
                '--bioactivities', '--structures',
            ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock_remove.call_args.kwargs['version'], ['2022.04.2', '2024.09.2'])
        self.assertTrue(mock_remove.call_args.kwargs['bioactivities'])
        self.assertTrue(mock_remove.call_args.kwargs['structures'])
        self.assertFalse(mock_remove.call_args.kwargs['proteins'])


class TestPdbmatchCommand(unittest.TestCase):

    def test_writes_matched_chunks_to_output(self):
        runner = CliRunner()
        chunk1, chunk2 = MagicMock(), MagicMock()
        with (
            patch('src.papyrus_scripts.cli.update_rcsb_data') as mock_update,
            patch('src.papyrus_scripts.cli.read_papyrus', return_value=MagicMock()) as mock_read,
            patch('src.papyrus_scripts.cli.get_num_rows_in_file', return_value=2_000_000) as mock_rows,
            patch('src.papyrus_scripts.cli.get_matches', return_value=[chunk1, chunk2]) as mock_matches,
            runner.isolated_filesystem(),
        ):
            result = runner.invoke(main, ['pdbmatch', '--output', 'out.tsv'])
        self.assertEqual(result.exit_code, 0, result.output)
        mock_update.assert_called_once()
        mock_read.assert_called_once()
        self.assertEqual(mock_matches.call_args.kwargs['total'], 2)
        chunk1.to_csv.assert_called_once_with('out.tsv', sep='\t', index=False, header=True, mode='w')
        chunk2.to_csv.assert_called_once_with('out.tsv', sep='\t', index=False, header=False, mode='a')
        # get_num_rows_in_file must use the same dataset variant as read_papyrus.
        self.assertEqual(mock_rows.call_args.kwargs['plusplus'], mock_read.call_args.kwargs['plusplus'])

    def test_more_flag_uses_full_dataset_row_count(self):
        runner = CliRunner()
        with (
            patch('src.papyrus_scripts.cli.update_rcsb_data'),
            patch('src.papyrus_scripts.cli.read_papyrus', return_value=MagicMock()) as mock_read,
            patch('src.papyrus_scripts.cli.get_num_rows_in_file', return_value=2_000_000) as mock_rows,
            patch('src.papyrus_scripts.cli.get_matches', return_value=[]),
            runner.isolated_filesystem(),
        ):
            result = runner.invoke(main, ['pdbmatch', '--output', 'out.tsv', '--more'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse(mock_read.call_args.kwargs['plusplus'])
        self.assertFalse(mock_rows.call_args.kwargs['plusplus'])


class TestFpsubsim2MutexOption(unittest.TestCase):
    """--output is required unless --fhelp is set; the two are mutually exclusive."""

    def test_output_and_fhelp_together_raise(self):
        runner = CliRunner()
        with patch('src.papyrus_scripts.cli.FPSubSim2'):
            result = runner.invoke(main, ['fpsubsim2', '--output', 'foo.h5', '--fhelp'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('mutually exclusive', result.output)

    def test_fhelp_alone_does_not_require_output(self):
        runner = CliRunner()
        with patch('src.papyrus_scripts.cli.FPSubSim2'):
            result = runner.invoke(main, ['fpsubsim2', '--fhelp'])
        self.assertEqual(result.exit_code, 0, result.output)


class TestFpsubsim2FingerprintHelp(unittest.TestCase):

    def test_prints_fingerprint_signatures(self):
        runner = CliRunner()
        with patch('src.papyrus_scripts.cli.FPSubSim2'):
            result = runner.invoke(main, ['fpsubsim2', '--fhelp'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('Morgan', result.output)
        self.assertIn('radius', result.output)


class TestFpsubsim2OutputNone(unittest.TestCase):

    def test_output_none_is_treated_as_no_output_file(self):
        runner = CliRunner()
        with patch('src.papyrus_scripts.cli.FPSubSim2') as mock_cls:
            result = runner.invoke(main, [
                'fpsubsim2', '--output', 'None', '--version', '2022.04.2', '--fingerprint', 'none',
            ])
        self.assertEqual(result.exit_code, 0, result.output)
        create = mock_cls.return_value.create_from_papyrus
        self.assertIsNone(create.call_args.kwargs['outfile'])


class TestFpsubsim2FingerprintParsing(unittest.TestCase):

    def _invoke(self, *args):
        runner = CliRunner()
        with patch('src.papyrus_scripts.cli.FPSubSim2') as mock_cls:
            result = runner.invoke(main, ['fpsubsim2', '--output', 'foo.h5',
                                          '--version', '2022.04.2', *args])
        return result, mock_cls.return_value.create_from_papyrus

    def test_fingerprint_without_parameters(self):
        result, create = self._invoke('--fingerprint', 'MACCSKeys')
        self.assertEqual(result.exit_code, 0, result.output)
        fps = create.call_args.kwargs['fingerprint']
        self.assertEqual(fps[0].name, 'MACCSKeys')

    def test_fingerprint_with_parameters(self):
        result, create = self._invoke('--fingerprint', 'Morgan;radius=3')
        self.assertEqual(result.exit_code, 0, result.output)
        fps = create.call_args.kwargs['fingerprint']
        self.assertEqual(fps[0].params['radius'], 3)

    def test_unknown_fingerprint_name_exits(self):
        result, create = self._invoke('--fingerprint', 'NotAFingerprint')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('must be one of', result.output)
        create.assert_not_called()

    def test_unknown_parameter_name_prints_warning(self):
        # cli.py prints a warning for an unrecognised parameter name but
        # still forwards it to the fingerprint constructor, which then
        # raises TypeError - a pre-existing bug, not something this test
        # suite is fixing; asserting the actual (crashing) behaviour here.
        result, create = self._invoke('--fingerprint', 'Morgan;bogus=1')
        self.assertIn('Parameters for fingerprint Morgan are', result.output)
        self.assertIsInstance(result.exception, TypeError)
        create.assert_not_called()

    def test_invalid_parameter_literal_exits(self):
        result, create = self._invoke('--fingerprint', 'Morgan;radius=not_a_literal')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('not a valid Python literal', result.output)
        create.assert_not_called()


class TestConvertCommand(unittest.TestCase):

    def _run(self, files, *args):
        """files: {relative_name: b''} written under a fake version directory."""
        runner = CliRunner()
        with runner.isolated_filesystem() as tmp:
            version_dir = Path(tmp) / 'version'
            version_dir.mkdir()
            (version_dir / 'subdir').mkdir()  # rglob('*') must skip non-file entries
            for name in files:
                (version_dir / name).write_bytes(b'')
            with (
                patch('src.papyrus_scripts.cli.process_data_version', return_value=MagicMock()),
                patch('src.papyrus_scripts.cli.papyrus_version_module') as mock_module,
                patch('src.papyrus_scripts.cli.convert_xz_to_gz') as mock_xz_to_gz,
                patch('src.papyrus_scripts.cli.convert_gz_to_xz') as mock_gz_to_xz,
            ):
                mock_module.return_value.base = version_dir
                result = runner.invoke(main, ['convert', *args])
        return result, mock_xz_to_gz, mock_gz_to_xz

    def test_infers_gzip_when_more_xz_than_gz(self):
        result, to_gz, to_xz = self._run({'a.xz': b'', 'b.xz': b'', 'c.gz': b''})
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(to_gz.call_count, 2)
        to_xz.assert_not_called()

    def test_infers_xz_when_more_gz_than_xz(self):
        result, to_gz, to_xz = self._run({'a.gz': b'', 'b.gz': b'', 'c.xz': b''})
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(to_xz.call_count, 2)
        to_gz.assert_not_called()

    def test_no_files_without_explicit_format_raises(self):
        result, to_gz, to_xz = self._run({})
        self.assertNotEqual(result.exit_code, 0)
        to_gz.assert_not_called()
        to_xz.assert_not_called()

    def test_explicit_format_skips_inference(self):
        result, to_gz, to_xz = self._run({'a.xz': b'', 'a.gz': b''}, '--format', 'gzip')
        self.assertEqual(result.exit_code, 0, result.output)
        to_gz.assert_called_once()
        to_xz.assert_not_called()

    def test_missing_xz_with_parquet_present_raises(self):
        result, to_gz, to_xz = self._run({'a.parquet': b''}, '--format', 'gzip')
        self.assertNotEqual(result.exit_code, 0)
        self.assertIsInstance(result.exception, FileNotFoundError)


class TestTestRealDataCommand(unittest.TestCase):

    def test_pytest_not_installed_raises_usage_error(self):
        runner = CliRunner()
        with patch.dict(sys.modules, {'pytest': None}):
            result = runner.invoke(main, ['test-real-data'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('pytest is required', result.output)

    def test_missing_test_file_raises_usage_error(self):
        runner = CliRunner()
        with patch.object(Path, 'is_file', return_value=False):
            result = runner.invoke(main, ['test-real-data'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('not found', result.output)

    def test_download_flag_calls_download_papyrus(self):
        runner = CliRunner()
        fake_pv = MagicMock(pystow_path_key='2024.09.2')
        with (
            patch('src.papyrus_scripts.cli.download_papyrus') as mock_download,
            patch('src.papyrus_scripts.cli.process_data_version', return_value=fake_pv),
            patch('pytest.main', return_value=0) as mock_pytest_main,
            patch.dict(os.environ, {}, clear=False),
        ):
            result = runner.invoke(main, ['test-real-data', '--download'])
            self.assertEqual(result.exit_code, 0, result.output)
            mock_download.assert_called_once()
            mock_pytest_main.assert_called_once()

    def test_sets_env_vars_from_options(self):
        runner = CliRunner()
        fake_pv = MagicMock(pystow_path_key='2024.09.2')
        with (
            patch('src.papyrus_scripts.cli.process_data_version', return_value=fake_pv),
            patch('pytest.main', return_value=0) as mock_pytest_main,
            patch.dict(os.environ, {}, clear=False),
        ):
            result = runner.invoke(main, [
                'test-real-data', '--indir', '/some/dir', '--sample-size', '5', '--verbose',
            ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(os.environ['PAPYRUS_REAL_DATA_VERSION'], '2024.09.2')
            self.assertEqual(os.environ['PAPYRUS_REAL_DATA_ROOT'], '/some/dir')
            self.assertEqual(os.environ['PAPYRUS_REAL_DATA_SAMPLE_SIZE'], '5')
            self.assertIn('-v', mock_pytest_main.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
