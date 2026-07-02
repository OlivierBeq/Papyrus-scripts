# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.subsim_search.FPSubSim2._parallel_create.

multiprocessing.Process/Queue are mocked entirely: no real worker processes
are spawned, no .h5 file or SD file is touched. FPSubSim2.__init__ is
bypassed (via __new__) since it requires optional deps (tables, FPSim2)
that aren't installed in this environment; the module itself imports fine
without them (the dependency check only runs on instantiation).
"""

import unittest
from unittest.mock import MagicMock, patch

from src.papyrus_scripts.subsim_search import FPSubSim2


def make_engine():
    engine = FPSubSim2.__new__(FPSubSim2)
    engine.sd_file = 'fake_sd_file.sdf'
    engine.h5_filename = 'fake_output.h5'
    return engine


class TestParallelCreateWorkerCount(unittest.TestCase):
    """Regression test: n_workers = multiprocessing.cpu_count() - 2 (njobs=-1
    branch) had no floor, unlike the njobs>=0 branch's max(njobs - 1, 1).
    On a <=2-core machine this produced 0 (or negative) workers, and with
    no worker ever draining the reader's queue, _reader_process's
    back-pressure guard slept forever - an indefinite hang.
    """

    def _run_with_mocked_multiprocessing(self, njobs, cpu_count):
        engine = make_engine()
        process_instances = []

        def fake_process(target=None, args=None, **kwargs):
            proc = MagicMock()
            proc.start = MagicMock()
            proc.join = MagicMock()
            proc.is_alive = MagicMock(return_value=False)
            process_instances.append((target, proc))
            return proc

        with patch('src.papyrus_scripts.subsim_search.multiprocessing.cpu_count', return_value=cpu_count), \
             patch('src.papyrus_scripts.subsim_search.multiprocessing.Process', side_effect=fake_process), \
             patch('src.papyrus_scripts.subsim_search.multiprocessing.Queue', return_value=MagicMock()), \
             patch('src.papyrus_scripts.subsim_search.sort_db_file'):
            engine._parallel_create(njobs, fingerprint=[], progress=False, total=None)

        return process_instances

    def test_njobs_minus_one_never_spawns_zero_workers_on_single_core(self):
        instances = self._run_with_mocked_multiprocessing(njobs=-1, cpu_count=1)
        worker_count = sum(
            1 for target, _ in instances
            if target is not None and target.__name__ == '_worker_process'
        )
        self.assertGreaterEqual(worker_count, 1)

    def test_njobs_minus_one_never_spawns_zero_workers_on_dual_core(self):
        instances = self._run_with_mocked_multiprocessing(njobs=-1, cpu_count=2)
        worker_count = sum(
            1 for target, _ in instances
            if target is not None and target.__name__ == '_worker_process'
        )
        self.assertGreaterEqual(worker_count, 1)


if __name__ == '__main__':
    unittest.main()
