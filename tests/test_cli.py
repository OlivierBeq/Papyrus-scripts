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


if __name__ == '__main__':
    unittest.main()
