# -*- coding: utf-8 -*-

"""Unit tests for the `download` command's --force flag: maps to
download_papyrus's disk_margin=0.0, vs. 0.1 by default.
"""

import unittest
from unittest.mock import patch

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


if __name__ == '__main__':
    unittest.main()
