# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.utils.IO.

These tests avoid network access and real Papyrus downloads: pystow's home
directory is redirected to a temporary folder, and offline fixture files
(the ones shipped with the package) are used for version/alias resolution.
"""

import json
import lzma
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from src.papyrus_scripts.utils import IO


class TestConvertXzToParquet(unittest.TestCase):
    """convert_xz_to_parquet reads with pandas.read_csv(chunksize=...,
    compression='xz') and writes each chunk via a streaming
    pyarrow.parquet.ParquetWriter - no intermediate decompressed file ever
    touches disk. Replaced an earlier implementation that decompressed to a
    temp file first and used Polars' scan_csv(...).sink_parquet(...)
    (falling back to splitting into row-bounded pieces above 2 GiB when
    even that wasn't enough): benchmarked on an 8M-row/3.3 GB-decompressed
    synthetic file, the old approach peaked at 3.2 GB RSS / 6.7 GB extra
    disk vs. 379 MB RSS / 50 MB extra disk here, and separately OOM-killed
    a real 39-47 GB Papyrus 3D bioactivity file (version 2024.09.1)
    outright on a 31 GB-RAM machine - this implementation has no such
    failure mode since it never materialises more than `chunksize` rows.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_unschema_dcolumn_with_late_na_does_not_crash(self):
        # Regression test: real Papyrus releases have shipped columns
        # absent from data_types.json (e.g. a "Year" column on the 3D
        # bioactivity file of version 2024.09.1, holding literal "NA" for
        # some rows). Unlike Polars, pandas has no nullable-by-default
        # integer type - a numeric column containing NA/NaN upcasts to
        # float64 rather than crashing, which is the expected, documented
        # behavior here (not a bug): pass an explicit schema_overrides
        # entry (e.g. pl.Int64, mapped to pandas' nullable "Int64") for any
        # column that must stay integer-typed despite missing values.
        years = [str(2000 + i % 20) for i in range(150)] + ['NA'] + ['2015'] * 10
        df = pl.DataFrame({
            'connectivity': [f'C{i}' for i in range(len(years))],
            'Year': years,
        })
        xz_path = Path(self._tmpdir.name) / 'test.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())

        out_path = Path(self._tmpdir.name) / 'test.parquet'
        # schema_overrides covers 'connectivity' only, mirroring a real
        # data_types.json that doesn't yet know about 'Year'.
        IO.convert_xz_to_parquet(
            xz_path, out_path, separator='\t',
            schema_overrides={'connectivity': pl.Utf8},
        )
        result = pl.read_parquet(out_path)
        self.assertEqual(result.height, len(years))
        self.assertEqual(result.schema['Year'], pl.Float64)
        expected = [None if y == 'NA' else float(y) for y in years]
        self.assertEqual(result['Year'].to_list(), expected)

    def test_schema_override_keeps_column_integer_despite_nulls(self):
        # The nullable-dtype counterpart of the test above: an explicit
        # pl.Int64 override maps to pandas' nullable "Int64" dtype, so the
        # column stays integer-typed (with real nulls) instead of
        # upcasting to float64.
        years = [str(2000 + i % 20) for i in range(150)] + ['NA'] + ['2015'] * 10
        df = pl.DataFrame({
            'connectivity': [f'C{i}' for i in range(len(years))],
            'Year': years,
        })
        xz_path = Path(self._tmpdir.name) / 'test_int.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        out_path = Path(self._tmpdir.name) / 'test_int.parquet'
        IO.convert_xz_to_parquet(
            xz_path, out_path, separator='\t',
            schema_overrides={'connectivity': pl.Utf8, 'Year': pl.Int64},
        )
        result = pl.read_parquet(out_path)
        self.assertEqual(result.schema['Year'], pl.Int64)
        expected = [None if y == 'NA' else int(y) for y in years]
        self.assertEqual(result['Year'].to_list(), expected)

    def test_cross_chunk_dtype_drift_is_healed_by_forcing_string(self):
        # A column not covered by schema_overrides can be inferred with a
        # different uniform dtype in different chunks - here, with
        # null_values=[] (so "NA" stays a literal string) and a tiny
        # chunksize, one chunk's Year column is all-numeric (infers int64)
        # while another holds the "NA" string (infers object/string).
        # ParquetWriter would reject writing the second chunk against the
        # first chunk's schema; convert_xz_to_parquet detects this before
        # that happens and restarts the whole conversion with Year forced
        # to 'string' from the first row, so it succeeds instead of raising.
        years = [str(2000 + i % 20) for i in range(150)] + ['NA'] + ['2015'] * 10
        df = pl.DataFrame({
            'connectivity': [f'C{i}' for i in range(len(years))],
            'Year': years,
        })
        xz_path = Path(self._tmpdir.name) / 'drift.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        out_path = Path(self._tmpdir.name) / 'drift.parquet'

        IO.convert_xz_to_parquet(
            xz_path, out_path, separator='\t',
            schema_overrides={'connectivity': pl.Utf8}, null_values=[],
            chunksize=50,
        )
        result = pl.read_parquet(out_path)
        self.assertEqual(result.height, len(years))
        self.assertEqual(result.schema['Year'], pl.Utf8)
        self.assertEqual(result['Year'].to_list(), years)
        # No stray temp files left behind.
        self.assertEqual(list(Path(self._tmpdir.name).glob('drift.parquet*')), [out_path])

    def test_failed_conversion_leaves_no_file_at_output_path(self):
        # Regression test: a conversion that OOM-killed partway through
        # (real incident on the 3D bioactivity file of 2024.09.1) left a
        # truncated, invalid file sitting at output_file - a hard kill
        # skips all Python try/finally cleanup, so writing must happen
        # under a different name, only renamed into place on success.
        # A hard kill can't be simulated directly, so a write failure is
        # forced deterministically instead, on the second chunk written.
        df = pl.DataFrame({
            'connectivity': [f'C{i}' for i in range(120)],
            'Year': [2000 + i % 20 for i in range(120)],
        })
        xz_path = Path(self._tmpdir.name) / 'crash.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        out_path = Path(self._tmpdir.name) / 'crash.parquet'

        real_write_table = pq.ParquetWriter.write_table
        calls = []

        def flaky_write_table(self, table, **kwargs):
            calls.append(table)
            if len(calls) == 2:
                raise OSError('simulated failure mid-write')
            return real_write_table(self, table, **kwargs)

        with (
            mock.patch.object(pq.ParquetWriter, 'write_table', flaky_write_table),
            self.assertRaises(OSError),
        ):
            IO.convert_xz_to_parquet(
                xz_path, out_path, separator='\t',
                schema_overrides={'connectivity': pl.Utf8, 'Year': pl.Int64},
                chunksize=50,
            )
        self.assertFalse(out_path.exists())
        # No stray temp files left behind either.
        self.assertEqual(list(Path(self._tmpdir.name).glob('crash.parquet*')), [])

    def test_null_values_override_disables_na_default(self):
        # Explicit null_values=[] must be respected as-is (e.g. so 'NA'
        # stays a literal string rather than becoming null).
        df = pl.DataFrame({'a': ['1', '2'], 'label': ['NA', 'ok']})
        xz_path = Path(self._tmpdir.name) / 'test2.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        out_path = Path(self._tmpdir.name) / 'test2.parquet'
        IO.convert_xz_to_parquet(xz_path, out_path, separator='\t', null_values=[])
        result = pl.read_parquet(out_path)
        self.assertEqual(result['label'].to_list(), ['NA', 'ok'])

    def test_legitimately_quoted_multiline_field_is_reconstructed_not_split(self):
        # Regression test for a bad "fix" that shipped and was reverted:
        # some Papyrus columns (InChI_AuxInfo, doc_id/citation fields)
        # legitimately hold values with an embedded literal '"' and even a
        # literal newline, properly RFC4180-quoted (doubled "" for a literal
        # quote, wrapped in "..." to allow embedded newlines) by the
        # exporter. A prior change disabled quote handling entirely,
        # believing stray quotes were corrupting row counts - confirmed on
        # the real Papyrus++ file of 2022.08.1 that this was backwards:
        # disabling quoting corrupted a genuine multi-line record
        # (BXNJHAXVSOCGBA_on_P68400_WT) into 3 garbage fragment rows
        # instead of the correct single reconstructed row. Default quoting
        # must stay enabled and must correctly reconstruct this pattern.
        text = (
            'connectivity\tvalue\n'
            'C1\tnormal\n'
            'C2\t"multi\nline value with a literal "" quote inside"\n'
            'C3\tnormal2\n'
        )
        xz_path = Path(self._tmpdir.name) / 'quotes.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(text.encode())
        out_path = Path(self._tmpdir.name) / 'quotes.parquet'

        IO.convert_xz_to_parquet(xz_path, out_path, separator='\t')
        result = pl.read_parquet(out_path)
        self.assertEqual(result.height, 3)
        self.assertEqual(result['connectivity'].to_list(), ['C1', 'C2', 'C3'])
        self.assertEqual(result['value'][1], 'multi\nline value with a literal " quote inside')

    def test_small_chunksize_matches_large_chunksize(self):
        # Every row is read and written chunk-by-chunk regardless of
        # chunksize - a small chunksize (forcing many chunks for this
        # ~500-row file) must produce output identical to a large one
        # (a single chunk). schema_overrides pins 'Year' to a nullable
        # dtype so per-chunk inference can't disagree between chunks.
        n = 500
        df = pl.DataFrame({
            'connectivity': [f'C{i}' for i in range(n)],
            'Year': [str(2000 + i % 20) if i != n // 2 else 'NA' for i in range(n)],
            'value': [i / 3 for i in range(n)],
        })
        xz_path = Path(self._tmpdir.name) / 'chunk_test.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        schema_overrides = {'connectivity': pl.Utf8, 'Year': pl.Int64, 'value': pl.Float64}

        large_chunk_out = Path(self._tmpdir.name) / 'large_chunk.parquet'
        IO.convert_xz_to_parquet(
            xz_path, large_chunk_out, separator='\t',
            schema_overrides=schema_overrides, chunksize=10_000,
        )
        large_chunk = pl.read_parquet(large_chunk_out)

        small_chunk_out = Path(self._tmpdir.name) / 'small_chunk.parquet'
        IO.convert_xz_to_parquet(
            xz_path, small_chunk_out, separator='\t',
            schema_overrides=schema_overrides, chunksize=17,
        )
        small_chunk = pl.read_parquet(small_chunk_out)

        self.assertEqual(small_chunk.schema, large_chunk.schema)
        self.assertTrue(small_chunk.equals(large_chunk))
        self.assertEqual(small_chunk.height, n)
        self.assertEqual(small_chunk['Year'].null_count(), 1)

    def test_progress_bar_options_do_not_affect_output(self):
        # total/leave/ncols only affect how the progress bar is displayed -
        # passing them (as download_papyrus does, with an approximate total
        # from data_size.json) must not change the converted data.
        n = 50
        df = pl.DataFrame({'connectivity': [f'C{i}' for i in range(n)]})
        xz_path = Path(self._tmpdir.name) / 'progress_opts.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        out_path = Path(self._tmpdir.name) / 'progress_opts.parquet'

        IO.convert_xz_to_parquet(
            xz_path, out_path, separator='\t',
            progress=True, total=n, leave=False, ncols=60,
        )
        result = pl.read_parquet(out_path)
        self.assertEqual(result.height, n)

    def test_int_schema_column_with_integral_floats_downcasts_to_int(self):
        # A column formatted with a trailing ".0" (e.g. "1.0") for every
        # value is still a real integer column - it must come out as such,
        # not stay float just because of how it happened to be formatted.
        df = pl.DataFrame({
            'connectivity': [f'C{i}' for i in range(120)],
            'count': [f'{i}.0' for i in range(120)],
        })
        xz_path = Path(self._tmpdir.name) / 'int_as_float_text.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        out_path = Path(self._tmpdir.name) / 'int_as_float_text.parquet'

        IO.convert_xz_to_parquet(
            xz_path, out_path, separator='\t',
            schema_overrides={'connectivity': pl.Utf8, 'count': pl.Int64},
            chunksize=50,
        )
        result = pl.read_parquet(out_path)
        self.assertEqual(result.schema['count'], pl.Int64)
        self.assertEqual(result['count'].to_list(), list(range(120)))

    def test_int_schema_column_with_genuine_fraction_does_not_crash(self):
        # Regression test: a real Papyrus release has shipped a column
        # documented as integer in data_types.json that actually holds
        # fraction-valued data for that release. Handing pandas the strict
        # nullable integer dtype directly for such a column crashes its C
        # parser outright ("TypeError: cannot safely cast non-equivalent
        # float64 to int64") the instant it reaches such a value - the
        # column must instead come out as Float64, losslessly.
        df = pl.DataFrame({
            'connectivity': [f'C{i}' for i in range(120)],
            'score': [i + 0.5 for i in range(120)],
        })
        xz_path = Path(self._tmpdir.name) / 'genuine_fraction.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        out_path = Path(self._tmpdir.name) / 'genuine_fraction.parquet'

        IO.convert_xz_to_parquet(
            xz_path, out_path, separator='\t',
            schema_overrides={'connectivity': pl.Utf8, 'score': pl.Int64},
            chunksize=50,
        )
        result = pl.read_parquet(out_path)
        self.assertEqual(result.schema['score'], pl.Float64)
        self.assertEqual(result['score'].to_list(), [i + 0.5 for i in range(120)])

    def test_int_schema_fraction_appearing_in_a_later_chunk_is_healed(self):
        # Same as above, but the fraction only appears well past the first
        # chunk - by the time it's discovered, earlier chunks have already
        # been written to the ParquetWriter with 'score' as Int64, so the
        # whole conversion must restart with it forced to Float64 from the
        # first row, exactly like the existing inferred-dtype drift healing.
        values = [float(i) for i in range(120)]
        values[100] = 100.5
        df = pl.DataFrame({
            'connectivity': [f'C{i}' for i in range(120)],
            'score': values,
        })
        xz_path = Path(self._tmpdir.name) / 'late_fraction.tsv.xz'
        with lzma.open(xz_path, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        out_path = Path(self._tmpdir.name) / 'late_fraction.parquet'

        IO.convert_xz_to_parquet(
            xz_path, out_path, separator='\t',
            schema_overrides={'connectivity': pl.Utf8, 'score': pl.Int64},
            chunksize=50,
        )
        result = pl.read_parquet(out_path)
        self.assertEqual(result.schema['score'], pl.Float64)
        self.assertEqual(result['score'].to_list(), values)


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
        # No revision given: falls back to the latest (here: only) revision
        # for that alias.
        v = IO.PapyrusVersion(version='2022.04')
        self.assertEqual(v.version_old_fmt, '05.4')

    def test_resolve_by_old_format_version_does_not_warn(self):
        # Regression test: every old-format string maps to exactly one
        # revision, so selecting it isn't resolving any real ambiguity -
        # it used to warn "revision not provided" regardless, which fired
        # on every call that resolves an already-downloaded old-format
        # version (e.g. via get_downloaded_versions), not just on genuinely
        # underspecified input.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            IO.PapyrusVersion(version='05.4')
        self.assertEqual(caught, [])

    def test_resolve_by_new_format_alias_does_not_warn_when_unambiguous(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            v = IO.PapyrusVersion(version='2022.04')
        self.assertEqual(v.version_old_fmt, '05.4')
        self.assertEqual(caught, [])

    def test_resolve_by_alias_warns_when_multiple_revisions_exist(self):
        # Simulate a genuinely ambiguous alias (two known revisions) - not
        # something the shipped aliases.json contains today, but the warning
        # exists for when it eventually does.
        original = IO.PapyrusVersion.aliases
        extra = original.iloc[[0]].copy()
        extra['revision'] = '99'
        IO.PapyrusVersion.aliases = pd.concat([original, extra], ignore_index=True)
        try:
            alias = original.iloc[0]['alias']
            with self.assertWarns(UserWarning):
                IO.PapyrusVersion(version=alias)
        finally:
            IO.PapyrusVersion.aliases = original

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

    def test_str_is_just_the_version_number(self):
        # Regression test: PapyrusVersion had no __str__, so any f-string/
        # str() use (progress bar labels, warnings, confirmation prompts)
        # fell back to __repr__'s "<PapyrusVersion version=...>" instead of
        # a clean, user-facing version string.
        v = IO.PapyrusVersion(version='05.4')
        self.assertEqual(str(v), '2022.04.2')

    def test_three_part_version_with_known_revision_resolves(self):
        v = IO.PapyrusVersion(version='2022.11.4')
        self.assertEqual(v.version, '2022.11.4')

    def test_three_part_version_with_out_of_range_revision_raises(self):
        # Regression test: the guard used to compare revisions as plain
        # strings ('4' < '130' is False lexicographically, since '4' > '1'),
        # so an out-of-range revision like 130 for alias 2022.11 (max
        # revision 4) was silently accepted instead of raising.
        with self.assertRaises(ValueError) as ctx:
            IO.PapyrusVersion(version='2022.11.130')
        self.assertIn('130', str(ctx.exception))
        self.assertIn('4', str(ctx.exception))

    def test_three_part_version_with_next_revision_raises(self):
        # Smaller, more realistic out-of-range case than the 130 example.
        with self.assertRaises(ValueError):
            IO.PapyrusVersion(version='2022.11.5')

    def test_double_digit_revision_compared_numerically_not_lexicographically(self):
        # Regression test: revision.max() on the string 'revision' column
        # used to pick the lexicographically greatest string ('9' > '10'),
        # not the numerically greatest one. Simulate an alias that has
        # reached a double-digit revision and check every code path that
        # calls max()/comparisons on 'revision' resolves to '10', not '9'.
        original = IO.PapyrusVersion.aliases
        alias = original.iloc[0]['alias']
        extra = original.iloc[[0]].copy()
        extra['revision'] = '10'
        patched = pd.concat([original, extra], ignore_index=True)
        IO.PapyrusVersion.aliases = patched
        try:
            # Resolving by alias alone (no revision) should pick '10', not '9'.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                v = IO.PapyrusVersion(version=alias)
            self.assertEqual(v.revision, '10')

            # The 3-part fallback path should accept revision 10 (not reject
            # it as out-of-range) and still reject revision 11.
            v10 = IO.PapyrusVersion(version=f'{alias}.10')
            self.assertEqual(v10.revision, '10')
            with self.assertRaises(ValueError):
                IO.PapyrusVersion(version=f'{alias}.11')

            # is_latest must also treat '10' as greater than '9'.
            if alias == patched['alias'].max():
                self.assertTrue(v10.is_latest)
        finally:
            IO.PapyrusVersion.aliases = original

    def test_resolve_latest_picks_numerically_greatest_revision(self):
        # Regression test: 'latest' resolution used revision.max() on the
        # string column for the winning alias; a double-digit revision (e.g.
        # '10') must beat a single-digit one (e.g. '9'), not lose to it
        # lexicographically.
        original = IO.PapyrusVersion.aliases
        latest_alias = original['alias'].max()
        latest_row = original[original['alias'] == latest_alias].iloc[[0]].copy()
        extra = latest_row.copy()
        extra['revision'] = '10'
        patched = pd.concat([original, extra], ignore_index=True)
        IO.PapyrusVersion.aliases = patched
        try:
            v = IO.PapyrusVersion(version='latest')
            self.assertEqual(v.revision, '10')
        finally:
            IO.PapyrusVersion.aliases = original


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