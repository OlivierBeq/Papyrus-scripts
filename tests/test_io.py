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
import warnings
from pathlib import Path

import polars as pl

from src.papyrus_scripts.utils import IO


class TestDataTypeNameStringSchema(unittest.TestCase):
    """Regression test: every data_types.json Papyrus actually ships uses
    plain lowercase type-name strings (e.g. "float", "int"), never the
    TypeEncoder/__type__-hydrated form TypeDecoder produces. to_polars_dtype
    only recognised real Python type objects (str, float, numpy dtypes, ...)
    - a plain string like "float" matched neither _BUILTIN_TO_POLARS'
    type-object keys nor any numpy type, so it silently fell through to the
    `return pl.Utf8` default regardless of its actual meaning. This forced
    *every* schema-driven column (every Papyrus++ column, every descriptor
    column read via read_molecular_descriptors/read_protein_descriptors) to
    String on read, since data_types.json is quoted in that plain-string
    format for 100% of real Papyrus releases.
    """

    def test_string_type_names_map_to_correct_dtypes(self):
        self.assertEqual(IO.to_polars_dtype('str'), pl.Utf8)
        self.assertEqual(IO.to_polars_dtype('object'), pl.Utf8)
        self.assertEqual(IO.to_polars_dtype('float'), pl.Float64)
        self.assertEqual(IO.to_polars_dtype('int'), pl.Int64)
        self.assertEqual(IO.to_polars_dtype('bool'), pl.Boolean)

    def test_unrecognised_string_falls_back_to_utf8(self):
        self.assertEqual(IO.to_polars_dtype('some_unknown_type'), pl.Utf8)

    def test_real_python_types_still_supported(self):
        # Must not regress the pre-existing type-object path (still used by
        # TypeDecoder-hydrated __type__ markers, and NumPy dtypes).
        import numpy as np
        self.assertEqual(IO.to_polars_dtype(float), pl.Float64)
        self.assertEqual(IO.to_polars_dtype(np.float32), pl.Float32)

    def test_to_polars_schema_maps_a_full_column_dict(self):
        schema = IO.to_polars_schema({
            'Activity_ID': 'str', 'pchembl_value_Mean': 'float', 'N': 'int',
        })
        self.assertEqual(schema, {
            'Activity_ID': pl.Utf8, 'pchembl_value_Mean': pl.Float64, 'N': pl.Int64,
        })

    def test_load_data_type_schemas_reads_real_shipped_format(self):
        with tempfile.TemporaryDirectory() as d:
            version_dir = Path(d) / 'papyrus' / '2022.04.2'
            version_dir.mkdir(parents=True)
            with open(version_dir / 'data_types.json', 'w') as fh:
                json.dump({
                    'papyrus': {'Activity_ID': 'str', 'pchembl_value_Mean': 'float'},
                    'ECFP6': {'connectivity': 'str', 'ECFP6_1': 'int'},
                }, fh)
            pv = IO.PapyrusVersion(version='2022.04.2')
            module = IO.papyrus_version_module(pv, root_folder=d)
            schemas = IO.load_data_type_schemas(module)
        self.assertEqual(schemas['papyrus']['pchembl_value_Mean'], pl.Float64)
        self.assertEqual(schemas['ECFP6']['ECFP6_1'], pl.Int64)


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
            path = Path(d) / 'data.json'
            IO.write_jsonfile({'x': 1, 'y': [1, 2, 3]}, path)
            result = IO.read_jsonfile(path)
            self.assertEqual(result, {'x': 1, 'y': [1, 2, 3]})

    def test_read_missing_file_returns_empty_dict(self):
        result = IO.read_jsonfile('/no/such/file.json')
        self.assertEqual(result, {})


class TestSha256(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.filepath = Path(self._tmpdir.name) / 'data.bin'
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
        self.dirpath = Path(self._tmpdir.name)
        for name in ('05.4_combined_set.tsv', '05.4_combined_set.tsv:ZoneIdentifier', 'other.tsv'):
            (self.dirpath / name).open('w').close()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_locate_matching_file(self):
        result = IO.locate_file(self.dirpath, r'\d+\.\d+_combined_set\.tsv.*')
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Path)
        self.assertEqual(result[0].name, '05.4_combined_set.tsv')

    def test_zone_identifier_files_are_ignored(self):
        result = IO.locate_file(self.dirpath, r'.*')
        self.assertTrue(all(not f.name.endswith(':ZoneIdentifier') for f in result))

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
    """These rely on the real (offline) aliases.json shipped with the package.

    Current released versions (old format -> alias.revision):
    05.4 -> 2022.04.2, 05.5 -> 2022.08.3, 05.6 -> 2022.11.4, 05.7 -> 2024.09.2
    (05.7/2024.09.2 is the only one with `pickett=True` and is the latest).
    """

    def test_resolve_by_old_format_version(self):
        v = IO.PapyrusVersion(version='05.4')
        self.assertEqual(v.version_old_fmt, '05.4')
        self.assertEqual(v.version, '2022.04.2')

    def test_resolve_by_new_format_alias(self):
        # No revision given: falls back to the latest revision for that alias
        # (and warns about it).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            v = IO.PapyrusVersion(version='2022.04')
        self.assertEqual(v.version_old_fmt, '05.4')

    def test_resolve_latest(self):
        v = IO.PapyrusVersion(version='latest')
        # 'latest' must resolve to the alphabetically/numerically greatest alias
        # combined with its greatest revision.
        aliases = IO.PapyrusVersion.aliases
        latest_alias = aliases['alias'].max()
        latest_rev = aliases[aliases['alias'] == latest_alias]['revision'].max()
        self.assertEqual(v.version, f'{latest_alias}.{latest_rev}')

    def test_resolve_by_chembl_version(self):
        v = IO.PapyrusVersion(chembl_version=29)
        self.assertEqual(v.version_old_fmt, '05.4')

    def test_resolve_by_multiple_source_flags(self):
        # Only one version includes both chembl and pickett data (05.7/2024.09.2).
        # Non-version/alias/revision flags are only exposed via `.params`, not as
        # direct attributes.
        v = IO.PapyrusVersion(chembl=True, pickett=True)
        self.assertTrue(v.params['pickett'])
        self.assertTrue(v.params['chembl'])
        self.assertEqual(v.version_old_fmt, '05.7')

    def test_unknown_version_raises(self):
        with self.assertRaises(ValueError):
            IO.PapyrusVersion(version='not_a_real_version')

    def test_ambiguous_query_raises(self):
        with self.assertRaises(ValueError):
            IO.PapyrusVersion()

    def test_repr(self):
        v = IO.PapyrusVersion(version='05.4')
        self.assertEqual(v.version_old_fmt, '05.4')
        self.assertIn('2022.04.2', repr(v))


class TestProcessDataVersion(unittest.TestCase):
    """process_data_version reads pystow's home directory, so PYSTOW_HOME is
    redirected to a throwaway tmp dir for the duration of each test.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_pystow_home = os.environ.get('PYSTOW_HOME')
        (Path(self._tmpdir.name) / 'papyrus').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self._old_pystow_home is None:
            os.environ.pop('PYSTOW_HOME', None)
        else:
            os.environ['PYSTOW_HOME'] = self._old_pystow_home
        self._tmpdir.cleanup()

    def _write_downloaded_versions(self, versions):
        path = Path(self._tmpdir.name) / 'papyrus' / 'versions.json'
        with open(path, 'w') as fh:
            json.dump(versions, fh)

    def test_no_downloaded_data_raises_io_error(self):
        with self.assertRaises(IOError):
            IO.process_data_version('05.4', root_folder=self._tmpdir.name)

    def test_downloaded_version_resolves(self):
        self._write_downloaded_versions(['05.4', '05.5'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
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

    def test_latest_falls_back_to_locally_downloaded_latest(self):
        # Regression test: process_data_version('latest') used to construct
        # PapyrusVersion(version='latest'), which resolves against the
        # *globally* known aliases table regardless of what's downloaded -
        # contradicting its own docstring's claim that 'latest' "resolves to
        # the newest downloaded version." Requesting 'latest' used to raise
        # ValueError whenever the globally newest release (05.7/2024.09.2)
        # hadn't been downloaded yet, even though older versions were present.
        self._write_downloaded_versions(['05.4', '05.5'])  # 05.7 is the true latest release
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            v = IO.process_data_version('latest', root_folder=self._tmpdir.name)
        self.assertEqual(v.version_old_fmt, '05.5')


class TestGetNumRowsInFile(unittest.TestCase):
    """get_num_rows_in_file reads pystow's home directory, so PYSTOW_HOME is
    redirected to a throwaway tmp dir for the duration of each test.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_pystow_home = os.environ.get('PYSTOW_HOME')
        self._version_dir = Path(self._tmpdir.name) / 'papyrus' / '05.4'
        self._version_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self._old_pystow_home is None:
            os.environ.pop('PYSTOW_HOME', None)
        else:
            os.environ['PYSTOW_HOME'] = self._old_pystow_home
        self._tmpdir.cleanup()

    def _write_data_size(self, sizes):
        path = self._version_dir / 'data_size.json'
        with open(path, 'w') as fh:
            json.dump(sizes, fh)

    def test_bioactivities_plusplus_canonical_key(self):
        self._write_data_size({'papyrus_++': 42})
        n = IO.get_num_rows_in_file(
            'bioactivities', is3D=False, version='05.4', plusplus=True,
            root_folder=self._tmpdir.name,
        )
        self.assertEqual(n, 42)

    def test_bioactivities_plusplus_legacy_key(self):
        self._write_data_size({'papyrus++': 42})
        n = IO.get_num_rows_in_file(
            'bioactivities', is3D=False, version='05.4', plusplus=True,
            root_folder=self._tmpdir.name,
        )
        self.assertEqual(n, 42)

    def test_bioactivities_plusplus_missing_key_raises(self):
        # Regression test: this branch used to silently return None via
        # sizes.get('papyrus_++', sizes.get('papyrus++')) when both keys
        # were absent, unlike every other branch in this function, which
        # uses direct indexing and raises a clear KeyError on a missing key.
        self._write_data_size({'papyrus_2D': 10})
        with self.assertRaises(KeyError):
            IO.get_num_rows_in_file(
                'bioactivities', is3D=False, version='05.4', plusplus=True,
                root_folder=self._tmpdir.name,
            )

    def test_root_folder_is_actually_used(self):
        # Regression test: get_num_rows_in_file called
        # papyrus_version_module(pv) without forwarding root_folder, whose
        # own default (root_folder=None) triggers _set_root_folder(None),
        # which *deletes* the PYSTOW_HOME env var this function had just set
        # two lines earlier - so data_size.json was always read from the
        # default pystow home, silently ignoring the caller's root_folder.
        self._write_data_size({'papyrus_2D': 7})
        n = IO.get_num_rows_in_file(
            'bioactivities', is3D=False, version='05.4', plusplus=False,
            root_folder=self._tmpdir.name,
        )
        self.assertEqual(n, 7)

    def test_structures_and_descriptors_respect_root_folder(self):
        self._write_data_size({
            'structures_2D': 5, 'structures_3D': 6,
            'mold2': 8, 'cddd': 9, 'ECFP6': 11, 'E3FP': 12,
            'mordred_2D': 13, 'mordred_3D': 14,
        })
        self.assertEqual(
            IO.get_num_rows_in_file('structures', is3D=False, version='05.4',
                                     root_folder=self._tmpdir.name),
            5,
        )
        self.assertEqual(
            IO.get_num_rows_in_file('descriptors', is3D=False, version='05.4',
                                     descriptor_name='mold2', root_folder=self._tmpdir.name),
            8,
        )


if __name__ == '__main__':
    unittest.main()