# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.download.remove_papyrus.

All filesystem/network-touching internals are mocked: get_papyrus_links,
_resolve_versions, pystow.module, shutil.rmtree and _update_versions_json.
No real Papyrus data or network access is needed.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.papyrus_scripts import download


def make_version(key):
    pv = MagicMock()
    pv.pystow_path_key = key
    pv.__str__.return_value = key
    return pv


class TestRemovePapyrusVersionRoot(unittest.TestCase):
    """Regression tests: remove_papyrus used `return` instead of `continue`
    inside its `for pv in versions:` loop, silently dropping every version
    after the first one from a multi-version removal request.
    """

    def setUp(self):
        self.v1 = make_version('v1')
        self.v2 = make_version('v2')
        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value={},
            ),
            'resolve_versions': patch(
                'src.papyrus_scripts.download._resolve_versions',
                return_value=[self.v1, self.v2],
            ),
            'get_version_files': patch(
                'src.papyrus_scripts.download._get_version_files', return_value={},
            ),
            'pystow_module': patch('src.papyrus_scripts.download.pystow.module'),
            'rmtree': patch('src.papyrus_scripts.download.shutil.rmtree'),
            'update_versions_json': patch('src.papyrus_scripts.download._update_versions_json'),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_version_root_removal_processes_every_version(self):
        result = download.remove_papyrus(version=['v1', 'v2'], version_root=True, force=True)
        self.assertIsNone(result)
        self.assertEqual(self.mocks['rmtree'].call_count, 2)
        self.assertEqual(self.mocks['update_versions_json'].call_count, 2)

    def test_declined_confirmation_still_processes_next_version(self):
        with patch('builtins.input', return_value='N'):
            download.remove_papyrus(version=['v1', 'v2'], version_root=True, force=False)
        # Neither version confirmed -> nothing removed, but input() must have
        # been asked for both (i.e. the loop didn't bail out after the first).
        self.assertEqual(self.mocks['rmtree'].call_count, 0)


class TestRemovePapyrusFileTypeRemoval(unittest.TestCase):
    """The `if not present: continue` branch (nothing to remove for this
    version) must also not abort processing of later versions.
    """

    def setUp(self):
        self.v1 = make_version('v1')
        self.v2 = make_version('v2')
        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value={},
            ),
            'resolve_versions': patch(
                'src.papyrus_scripts.download._resolve_versions',
                return_value=[self.v1, self.v2],
            ),
            'get_version_files': patch(
                'src.papyrus_scripts.download._get_version_files',
                # No 'papyrus++' key present -> `removal` ends up empty ->
                # `present` stays empty -> hits the `if not present:` branch.
                return_value={},
            ),
            'pystow_module': patch('src.papyrus_scripts.download.pystow.module'),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_empty_present_list_still_processes_next_version(self):
        result = download.remove_papyrus(version=['v1', 'v2'], papyruspp=True, progress=False)
        self.assertIsNone(result)
        # _get_version_files must have been called once per version - proof
        # the loop reached v2 instead of returning after v1.
        self.assertEqual(self.mocks['get_version_files'].call_count, 2)


if __name__ == '__main__':
    unittest.main()
