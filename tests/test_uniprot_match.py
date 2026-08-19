# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.utils.UniprotMatch.UniprotMatch.

Mocks the HTTP session entirely: no network access is needed.
"""

import contextlib
import gzip
import io
import unittest
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree

import pandas as pd
from requests.adapters import Retry

from src.papyrus_scripts.utils.UniprotMatch import UniprotMatch, uniprot_mappings


class TestCheckIdMappingResultsReady(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)
        self.mapper._session = MagicMock()

    def test_failed_job_raises_with_real_status(self):
        # Regression test: the failure branch indexed the raw
        # requests.Response object (`request["jobStatus"]`, not subscriptable)
        # instead of the parsed JSON dict `j`, so a genuine job failure
        # crashed with an opaque TypeError instead of surfacing the actual
        # jobStatus value.
        response = MagicMock()
        response.json.return_value = {'jobStatus': 'FAILED'}
        self.mapper._session.get.return_value = response

        with self.assertRaises(Exception) as ctx:
            self.mapper._check_id_mapping_results_ready('job1', verbose=False)
        self.assertEqual(str(ctx.exception), 'FAILED')

    def test_finished_job_returns_true(self):
        response = MagicMock()
        response.json.return_value = {'results': [{'from': 'A', 'to': 'B'}], 'failedIds': []}
        self.mapper._session.get.return_value = response

        self.assertTrue(self.mapper._check_id_mapping_results_ready('job1', verbose=False))

    def test_running_job_polls_then_returns(self):
        running = MagicMock()
        running.json.return_value = {'jobStatus': 'RUNNING'}
        ready = MagicMock()
        ready.json.return_value = {'results': [], 'failedIds': ['x']}
        self.mapper._session.get.side_effect = [running, ready]

        self.assertTrue(self.mapper._check_id_mapping_results_ready('job1', verbose=True))
        self.assertEqual(self.mapper._session.get.call_count, 2)


class TestUniprotMatchInit(unittest.TestCase):

    def test_custom_retry_builds_own_session(self):
        mapper = UniprotMatch(retry=Retry(total=1))
        self.assertIn('User-Agent', mapper._session.headers)


class TestSubmitIdMapping(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)
        self.mapper._session = MagicMock()
        response = MagicMock()
        response.json.return_value = {'jobId': 'JOB123'}
        self.mapper._session.post.return_value = response

    def test_gene_name_without_taxon_raises(self):
        with self.assertRaises(ValueError):
            self.mapper._submit_id_mapping('Gene_Name', 'ACC', ['a'], taxon=None)

    def test_without_taxon_posts_and_returns_job_id(self):
        job_id = self.mapper._submit_id_mapping('UniProtKB_AC-ID', 'PDB', ['a', 'b'], taxon=None)
        self.assertEqual(job_id, 'JOB123')
        _, kwargs = self.mapper._session.post.call_args
        self.assertNotIn('taxId', kwargs['data'])
        self.assertEqual(kwargs['data']['ids'], 'a,b')

    def test_with_taxon_posts_taxid(self):
        self.mapper._submit_id_mapping('Gene_Name', 'ACC', ['a'], taxon='9606')
        _, kwargs = self.mapper._session.post.call_args
        self.assertEqual(kwargs['data']['taxId'], '9606')


class TestGetNextLink(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)

    def test_single_relation_header(self):
        headers = {'Link': '<https://example.org/page2>; rel="next"'}
        self.assertEqual(self.mapper._get_next_link(headers), 'https://example.org/page2')

    def test_no_link_header_returns_none(self):
        self.assertIsNone(self.mapper._get_next_link({}))

    def test_multiple_relations_extracts_only_next(self):
        # A Link header may legally hold several comma-separated relations
        # (RFC 8288); the 'next' URL must be extracted correctly even when
        # 'rel="prev"' precedes it.
        headers = {
            'Link': '<https://example.org/page0>; rel="prev", '
                   '<https://example.org/page2>; rel="next"',
        }
        self.assertEqual(self.mapper._get_next_link(headers), 'https://example.org/page2')

    def test_next_before_prev_still_extracts_next(self):
        headers = {
            'Link': '<https://example.org/page2>; rel="next", '
                   '<https://example.org/page0>; rel="prev"',
        }
        self.assertEqual(self.mapper._get_next_link(headers), 'https://example.org/page2')


class TestGetBatch(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)
        self.mapper._session = MagicMock()

    def test_stops_when_no_next_link(self):
        first = MagicMock(headers={})
        batches = list(self.mapper._get_batch(first, 'json', False))
        self.assertEqual(batches, [])

    def test_yields_each_page_until_no_next_link(self):
        first = MagicMock(headers={'Link': '<https://x/page2>; rel="next"'})
        page2 = MagicMock(headers={})
        page2.json.return_value = {'results': ['b']}
        self.mapper._session.get.return_value = page2

        batches = list(self.mapper._get_batch(first, 'json', False))
        self.assertEqual(batches, [{'results': ['b']}])
        self.mapper._session.get.assert_called_once_with('https://x/page2')


class TestCombineBatches(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)

    def test_json_merges_results_and_failed_ids(self):
        all_results = {'results': ['a'], 'failedIds': []}
        batch = {'results': ['b'], 'failedIds': ['x']}
        merged = self.mapper._combine_batches(all_results, batch, 'json')
        self.assertEqual(merged, {'results': ['a', 'b'], 'failedIds': ['x']})

    def test_json_ignores_empty_batch_keys(self):
        all_results = {'results': ['a'], 'failedIds': []}
        batch = {'results': [], 'failedIds': []}
        merged = self.mapper._combine_batches(all_results, batch, 'json')
        self.assertEqual(merged, all_results)

    def test_tsv_skips_batch_header_row(self):
        merged = self.mapper._combine_batches(['h', 'r1'], ['h', 'r2'], 'tsv')
        self.assertEqual(merged, ['h', 'r1', 'r2'])

    def test_other_format_concatenates(self):
        merged = self.mapper._combine_batches([b'a'], [b'b'], 'xml')
        self.assertEqual(merged, [b'a', b'b'])


class TestGetIdMappingResultsLink(unittest.TestCase):

    def test_returns_redirect_url(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            mapper = UniprotMatch(polling_interval=0)
        mapper._session = MagicMock()
        response = MagicMock()
        response.json.return_value = {'redirectURL': 'https://x/results/JOB'}
        mapper._session.get.return_value = response

        self.assertEqual(mapper._get_id_mapping_results_link('JOB'), 'https://x/results/JOB')


class TestDecodeResults(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)

    def _gzip_response(self, content: bytes) -> MagicMock:
        response = MagicMock()
        response.content = gzip.compress(content)
        return response

    def test_compressed_json(self):
        response = self._gzip_response(b'{"a": 1}')
        self.assertEqual(self.mapper._decode_results(response, 'json', True), {'a': 1})

    def test_compressed_tsv_drops_empty_lines(self):
        response = self._gzip_response(b'h\tv\n\nr\tv2\n')
        self.assertEqual(self.mapper._decode_results(response, 'tsv', True), ['h\tv', 'r\tv2'])

    def test_compressed_xlsx(self):
        response = self._gzip_response(b'binarydata')
        self.assertEqual(self.mapper._decode_results(response, 'xlsx', True), [b'binarydata'])

    def test_compressed_xml(self):
        response = self._gzip_response(b'<a/>')
        self.assertEqual(self.mapper._decode_results(response, 'xml', True), ['<a/>'])

    def test_compressed_other_format(self):
        response = self._gzip_response(b'plain')
        self.assertEqual(self.mapper._decode_results(response, 'list', True), 'plain')

    def test_uncompressed_json(self):
        response = MagicMock()
        response.json.return_value = {'a': 1}
        self.assertEqual(self.mapper._decode_results(response, 'json', False), {'a': 1})

    def test_uncompressed_tsv_drops_empty_lines(self):
        response = MagicMock(text='h\tv\n\nr\tv2\n')
        self.assertEqual(self.mapper._decode_results(response, 'tsv', False), ['h\tv', 'r\tv2'])

    def test_uncompressed_xlsx(self):
        response = MagicMock(content=b'binarydata')
        self.assertEqual(self.mapper._decode_results(response, 'xlsx', False), [b'binarydata'])

    def test_uncompressed_xml(self):
        response = MagicMock(text='<a/>')
        self.assertEqual(self.mapper._decode_results(response, 'xml', False), ['<a/>'])

    def test_uncompressed_other_format(self):
        response = MagicMock(text='plain')
        self.assertEqual(self.mapper._decode_results(response, 'list', False), 'plain')


class TestXmlHelpers(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)

    def test_get_xml_namespace_present(self):
        element = ElementTree.fromstring('<a xmlns="http://x"/>')  # noqa: S314 - hardcoded literal, not untrusted input
        self.assertEqual(self.mapper._get_xml_namespace(element), 'http://x')

    def test_get_xml_namespace_absent(self):
        element = ElementTree.fromstring('<a/>')  # noqa: S314 - hardcoded literal, not untrusted input
        self.assertEqual(self.mapper._get_xml_namespace(element), '')

    def test_merge_xml_results_inserts_entries_before_last_child(self):
        xml1 = ('<uniprot xmlns="http://uniprot.org/uniprot">'
                '<entry>A</entry><copyright>c</copyright></uniprot>')
        xml2 = ('<uniprot xmlns="http://uniprot.org/uniprot">'
                '<entry>B</entry><copyright>c</copyright></uniprot>')

        merged = self.mapper._merge_xml_results([xml1, xml2])

        root = ElementTree.fromstring(merged)  # noqa: S314 - derived from hardcoded literals above, not untrusted input
        entries = root.findall('{http://uniprot.org/uniprot}entry')
        self.assertEqual([e.text for e in entries], ['A', 'B'])


class TestPrintProgressBatches(unittest.TestCase):

    def test_prints_fetched_count(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            mapper = UniprotMatch(polling_interval=0)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mapper._print_progress_batches(0, 2, 5)
        self.assertEqual(out.getvalue().strip(), 'Fetched: 2 / 5')

    def test_caps_at_total(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            mapper = UniprotMatch(polling_interval=0)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mapper._print_progress_batches(2, 2, 5)
        self.assertEqual(out.getvalue().strip(), 'Fetched: 5 / 5')


class TestGetIdMappingResultsSearch(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)
        self.mapper._session = MagicMock()

    def test_single_page_json_uses_default_size(self):
        # No 'size' in the query string - exercises the default-size branch.
        response = MagicMock(headers={'x-total-results': '2'})
        response.json.return_value = {'results': ['a', 'b'], 'failedIds': []}
        self.mapper._session.get.return_value = response

        result = self.mapper._get_id_mapping_results_search(
            'https://rest.uniprot.org/idmapping/results/JOB?format=json', verbose=True,
        )
        self.assertEqual(result, {'results': ['a', 'b'], 'failedIds': []})

    def test_paginated_json_uses_explicit_size(self):
        page1 = MagicMock(headers={
            'x-total-results': '4',
            'Link': '<https://rest.uniprot.org/next>; rel="next"',
        })
        page1.json.return_value = {'results': ['a', 'b'], 'failedIds': []}
        page2 = MagicMock(headers={'x-total-results': '4'})
        page2.json.return_value = {'results': ['c', 'd'], 'failedIds': []}
        self.mapper._session.get.side_effect = [page1, page2]

        result = self.mapper._get_id_mapping_results_search(
            'https://rest.uniprot.org/idmapping/results/JOB?format=json&size=2', verbose=True,
        )
        self.assertEqual(result, {'results': ['a', 'b', 'c', 'd'], 'failedIds': []})

    def test_xml_format_merges_pages(self):
        xml_page = ('<uniprot xmlns="http://uniprot.org/uniprot">'
                    '<entry>A</entry><copyright>c</copyright></uniprot>')
        response = MagicMock(headers={'x-total-results': '1'}, text=xml_page)
        self.mapper._session.get.return_value = response

        result = self.mapper._get_id_mapping_results_search(
            'https://rest.uniprot.org/idmapping/results/JOB?format=xml',
        )
        self.assertIn(b'<entry>A</entry>', result)


class TestGetIdMappingResultsStream(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)
        self.mapper._session = MagicMock()
        response = MagicMock()
        response.json.return_value = {'a': 1}
        self.mapper._session.get.return_value = response

    def test_replaces_results_with_stream(self):
        self.mapper._get_id_mapping_results_stream(
            'https://rest.uniprot.org/idmapping/results/JOB?format=json',
        )
        self.mapper._session.get.assert_called_once_with(
            'https://rest.uniprot.org/idmapping/stream/JOB?format=json',
        )

    def test_leaves_stream_url_untouched(self):
        url = 'https://rest.uniprot.org/idmapping/stream/JOB?format=json'
        self.mapper._get_id_mapping_results_stream(url)
        self.mapper._session.get.assert_called_once_with(url)


class TestUniprotIdMapping(unittest.TestCase):

    def setUp(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.new_session'):
            self.mapper = UniprotMatch(polling_interval=0)

    def test_returns_none_when_results_not_ready(self):
        with (
            patch.object(self.mapper, '_submit_id_mapping', return_value='JOB1'),
            patch.object(self.mapper, '_check_id_mapping_results_ready', return_value=False),
        ):
            self.assertIsNone(self.mapper.uniprot_id_mapping(['a']))

    def test_builds_mapping_for_string_and_dict_results(self):
        search_result = {'results': [
            {'from': 'Q1', 'to': 'PDB1'},
            {'from': 'Q1', 'to': 'PDB2'},
            {'from': 'Q2', 'to': {'primaryAccession': 'ACC2'}},
        ]}
        with (
            patch.object(self.mapper, '_submit_id_mapping', return_value='JOB1'),
            patch.object(self.mapper, '_check_id_mapping_results_ready', return_value=True),
            patch.object(self.mapper, '_get_id_mapping_results_link', return_value='https://x'),
            patch.object(self.mapper, '_get_id_mapping_results_search', return_value=search_result),
        ):
            result = self.mapper.uniprot_id_mapping(['Q1', 'Q2'], verbose=False)
        self.assertEqual(result, {'Q1': 'PDB1 PDB2', 'Q2': 'ACC2'})


class TestUniprotMappingsFunction(unittest.TestCase):

    def test_sifts_path_filters_by_query(self):
        sifts = pd.DataFrame({'SP_PRIMARY': ['Q1', 'Q2'], 'PDB': ['1abc;2xyz', '3foo']})
        with patch('src.papyrus_scripts.utils.UniprotMatch.pd.read_csv', return_value=sifts):
            result = uniprot_mappings('1abc', map_from='PDB', map_to='ACC')
        self.assertEqual(list(result['PDB']), ['1abc'])
        self.assertEqual(list(result['ACC']), ['Q1'])

    def test_sifts_path_no_query_returns_all(self):
        sifts = pd.DataFrame({'SP_PRIMARY': ['Q1'], 'PDB': ['1abc']})
        with patch('src.papyrus_scripts.utils.UniprotMatch.pd.read_csv', return_value=sifts):
            result = uniprot_mappings(None, map_from='PDB_ID', map_to='UniProtKB_AC-ID')
        self.assertEqual(len(result), 1)

    def test_api_path_uses_uniprot_match(self):
        with patch('src.papyrus_scripts.utils.UniprotMatch.UniprotMatch') as mock_cls:
            mock_cls.return_value.uniprot_id_mapping.return_value = {'Q1': 'X1'}
            result = uniprot_mappings('Q1', map_from='ID', map_to='PDB_ID')
        self.assertEqual(result.loc[0, 'ID'], 'Q1')
        self.assertEqual(result.loc[0, 'PDB_ID'], 'X1')


if __name__ == '__main__':
    unittest.main()
