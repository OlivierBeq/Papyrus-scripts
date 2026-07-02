# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.utils.UniprotMatch.UniprotMatch.

Mocks the HTTP session entirely: no network access is needed.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.papyrus_scripts.utils.UniprotMatch import UniprotMatch


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


if __name__ == '__main__':
    unittest.main()
