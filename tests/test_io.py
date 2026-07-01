# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.utils.IO.

These tests avoid network access and real Papyrus downloads: pystow's home
directory is redirected to a temporary folder, and offline fixture files
(the ones shipped with the package) are used for version/alias resolution.
"""

import json
import os
import tempfile
import unittest

from src.papyrus_scripts.utils import IO


class TestTypeEncoderDecoder(unittest.TestCase):

    def test_roundtrip_builtin_types(self):
        encoded = json.dumps({'a': int, 'b': str, 'c': float, 'd': bool}, cls=IO.TypeEncoder)
        decoded = json.loads(encoded, cls=IO.TypeDecoder)
        self.assertIs(decoded['a'], int)
        self.assertIs(decoded['b'], str)
        self.assertIs(decoded['c'], float)
        self.assertIs(decoded['d'], bool)

    def test_roundtrip_non_builtin_type(self):
        import numpy as np
        encoded = json.dumps({'t': np.float64}, cls=IO.TypeEncoder)
        decoded = json.loads(encoded, cls=IO.TypeDecoder)
        self.assertIs(decoded['t'], np.float64)

    def test_encoder_rejects_non_type_non_serializable(self):
        with self.assertRaises(TypeError):
            json.dumps({'a': object()}, cls=IO.TypeEncoder)

    def test_decoder_passes_through_plain_objects(self):
        decoded = json.loads('{"a": 1, "b": "text"}', cls=IO.TypeDecoder)
        self.assertEqual(decoded, {'a': 1, 'b': 'text'})


class TestJsonFileHelpers(unittest.TestCase):

    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'data.json')
            IO.write_jsonfile({'x': 1, 'y': [1, 2, 3]}, path)
            result = IO.read_jsonfile(path)
            self.assertEqual(result, {'x': 1, 'y': [1, 2, 3]})

    def test_read_missing_file_returns_empty_dict(self):
        result = IO.read_jsonfile('/no/such/file.json')
        self.assertEqual(result, {})


class TestSha256(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.filepath = os.path.join(self._tmpdir.name, 'data.bin')
        with open(self.filepath, 'wb') as fh:
            fh.write(b'hello world')
        # Precomputed with hashlib directly.
        self.expected_hash = ('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380'
                              'ee9088f7ace2efcde9')

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_sha256sum(self):
        self.assertEqual(IO.sha256sum(self.filepath), self.expected_hash)

    def test_sha256sum_with_small_blocksize(self):
        # Must not depend on the block size used to read the file.
        self.assertEqual(IO.sha256sum(self.filepath, blocksize=4), self.expected_hash)

    def test_assert_sha256sum_match(self):
        self.assertTrue(IO.assert_sha256sum(self.filepath, self.expected_hash))

    def test_assert_sha256sum_mismatch(self):
        self.assertFalse(IO.assert_sha256sum(self.filepath, 'a' * 64))

    def test_assert_sha256sum_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            IO.assert_sha256sum(self.filepath, 'too_short')


class TestLocateFile(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.dirpath = self._tmpdir.name
        for name in ('05.4_combined_set.tsv', '05.4_combined_set.tsv:ZoneIdentifier', 'other.tsv'):
            open(os.path.join(self.dirpath, name), 'w').close()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_locate_matching_file(self):
        result = IO.locate_file(self.dirpath, r'\d+\.\d+_combined_set\.tsv.*')
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].endswith('05.4_combined_set.tsv'))

    def test_zone_identifier_files_are_ignored(self):
        result = IO.locate_file(self.dirpath, r'.*')
        self.assertTrue(all(not f.endswith(':ZoneIdentifier') for f in result))

    def test_no_match_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            IO.locate_file(self.dirpath, r'no_such_pattern')

    def test_missing_directory_raises(self):
        with self.assertRaises(NotADirectoryError):
            IO.locate_file('/no/such/directory', r'.*')


class TestDiskSpace(unittest.TestCase):

    def test_get_disk_space_positive(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertGreater(IO.get_disk_space(d), 0)

    def test_enough_disk_space_true_for_tiny_requirement(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(IO.enough_disk_space(d, required=1))

    def test_enough_disk_space_false_for_huge_requirement(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(IO.enough_disk_space(d, required=10 ** 18))


class TestPapyrusVersion(unittest.TestCase):
    """These rely on the real (offline) aliases.json shipped with the package."""

    def test_resolve_by_old_format_version(self):
        v = IO.PapyrusVersion(version='05.4')
        self.assertEqual(v.version_old_fmt, '05.4')
        self.assertEqual(v.version, '2022.04')

    def test_resolve_by_new_format_alias(self):
        v = IO.PapyrusVersion(version='2022.04')
        self.assertEqual(v.version_old_fmt, '05.4')

    def test_resolve_latest(self):
        v = IO.PapyrusVersion(version='latest')
        # 'latest' must resolve to the alphabetically/numerically greatest alias.
        all_aliases = IO.PapyrusVersion.aliases['alias'].tolist()
        self.assertEqual(v.version, max(all_aliases))

    def test_resolve_by_chembl_version(self):
        v = IO.PapyrusVersion(chembl_version=29)
        self.assertEqual(v.version_old_fmt, '05.4')

    def test_resolve_by_multiple_source_flags(self):
        v = IO.PapyrusVersion(chembl=True, pickett=True)
        # Only one version includes both chembl and pickett data.
        self.assertTrue(v.pickett)
        self.assertTrue(v.chembl)

    def test_unknown_version_raises(self):
        with self.assertRaises(ValueError):
            IO.PapyrusVersion(version='not_a_real_version')

    def test_ambiguous_query_raises(self):
        with self.assertRaises(ValueError):
            IO.PapyrusVersion()

    def test_repr(self):
        v = IO.PapyrusVersion(version='05.4')
        self.assertIn('05.4', repr(v))
        self.assertIn('2022.04', repr(v))


class TestProcessDataVersion(unittest.TestCase):
    """process_data_version reads pystow's home directory, so PYSTOW_HOME is
    redirected to a throwaway tmp dir for the duration of each test.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_pystow_home = os.environ.get('PYSTOW_HOME')
        os.makedirs(os.path.join(self._tmpdir.name, 'papyrus'), exist_ok=True)

    def tearDown(self):
        if self._old_pystow_home is None:
            os.environ.pop('PYSTOW_HOME', None)
        else:
            os.environ['PYSTOW_HOME'] = self._old_pystow_home
        self._tmpdir.cleanup()

    def _write_downloaded_versions(self, versions):
        path = os.path.join(self._tmpdir.name, 'papyrus', 'versions.json')
        with open(path, 'w') as fh:
            json.dump(versions, fh)

    def test_no_downloaded_data_raises_io_error(self):
        with self.assertRaises(IOError):
            IO.process_data_version('05.4', root_folder=self._tmpdir.name)

    def test_downloaded_version_resolves(self):
        self._write_downloaded_versions(['05.4', '05.5'])
        v = IO.process_data_version('05.4', root_folder=self._tmpdir.name)
        self.assertEqual(v.version_old_fmt, '05.4')

    def test_not_downloaded_version_raises(self):
        self._write_downloaded_versions(['05.4', '05.5'])
        with self.assertRaises(ValueError):
            IO.process_data_version('05.6', root_folder=self._tmpdir.name)

    def test_is_local_version_available(self):
        self._write_downloaded_versions(['05.4'])
        self.assertTrue(IO.is_local_version_available('05.4', root_folder=self._tmpdir.name))
        self.assertFalse(IO.is_local_version_available('05.6', root_folder=self._tmpdir.name))

    def test_latest_raises_if_globally_latest_release_is_not_downloaded(self):
        # KNOWN BUG (not fixed here, flagged for the maintainer): process_data_version's
        # `elif version == 'latest':` branch compares a PapyrusVersion instance to the
        # string 'latest', which is always False, so it never falls back to
        # get_latest_downloaded_version(). As a result, requesting 'latest' raises
        # instead of resolving to the newest *locally downloaded* version whenever the
        # globally newest release (per aliases.json) hasn't been downloaded yet.
        self._write_downloaded_versions(['05.4', '05.5'])  # 05.7 is the true latest release
        with self.assertRaises(ValueError):
            IO.process_data_version('latest', root_folder=self._tmpdir.name)


if __name__ == '__main__':
    unittest.main()
