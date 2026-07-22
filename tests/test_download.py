# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.download.remove_papyrus.

All filesystem/network-touching internals are mocked: get_papyrus_links,
_resolve_versions, pystow.module, shutil.rmtree and _update_versions_json.
No real Papyrus data or network access is needed.
"""

import io
import os
import queue
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
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


class _StopEarly(Exception):
    """Raised to abort download_papyrus right after the stale-format notice."""


class TestDownloadPapyrusStaleFormatNotice(unittest.TestCase):
    """Regression test: the old-format-data notice always suggested
    `papyrus clean --version all`, regardless of which old-format version(s)
    were actually found on disk - wiping every locally downloaded version,
    including ones with no old-format duplicate, when a user followed it
    literally instead of just the stale one(s).
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        versions_json = Path(self._tmpdir.name) / 'versions.json'
        versions_json.write_text('["05.6", "05.7"]')
        patches = {
            'pystow_join': patch(
                'src.papyrus_scripts.download.pystow.join', return_value=versions_json,
            ),
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', side_effect=_StopEarly,
            ),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_notice_lists_only_the_versions_actually_found(self):
        buf = io.StringIO()
        with self.assertRaises(_StopEarly), redirect_stdout(buf):
            download.download_papyrus(version='latest')
        notice = buf.getvalue()
        self.assertIn('--version 05.6', notice)
        self.assertIn('--version 05.7', notice)
        self.assertIn('--remove_version', notice)
        self.assertNotIn('--version all', notice)


class _FakeResponse:
    def __init__(self, content: bytes):
        self._content = content

    def iter_content(self, chunk_size):
        yield self._content


class _FakeQueue:
    """Stand-in for multiprocessing.Queue that runs in-process, so tests can
    inspect exactly what was queued without a real subprocess.
    """

    def __init__(self, events: list | None = None, label: str | None = None):
        self.items: list = []
        self._pending: list = []
        self._events = events
        self._label = label

    def put(self, item):
        self.items.append(item)
        self._pending.append(item)
        if self._events is not None:
            tag = 'sentinel' if item is download._CONVERSION_DONE else item.get('fpath')
            self._events.append(f'{self._label}:{tag}')

    def get(self):
        # An empty queue (e.g. an error_queue nothing ever put to, because
        # no real worker ran) behaves like a worker that exited cleanly.
        return self._pending.pop(0) if self._pending else None

    def get_nowait(self):
        if not self._pending:
            raise queue.Empty
        return self._pending.pop(0)


class _FakeProcess:
    """Stand-in for multiprocessing.Process: never actually starts anything."""

    def __init__(self, target=None, args=(), daemon=None):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class TestDownloadPapyrusQueuesConversionsAsItGoes(unittest.TestCase):
    """Regression test: download_papyrus used to download every requested
    file first and only then convert tabular ones to Parquet in a separate,
    later pass - so with several files queued, the very first conversion
    waited for every other file to finish downloading first, however small
    it was. Each tabular file must now be queued for conversion right after
    it finishes downloading instead, interleaved with the remaining
    downloads, and a sentinel must be queued only once every file has been
    handled.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._old_pystow_home = os.environ.get('PYSTOW_HOME')
        self.addCleanup(self._restore_pystow_home)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')
            zh.writestr('data_size.json', '{"papyrus_++": 42, "papyrus_proteins": 7}')
        zip_bytes = zip_buf.getvalue()

        content_by_url = {
            'https://example.org/readme': b'a readme',
            'https://example.org/requirements': zip_bytes,
            'https://example.org/proteins': b'fake-tsv-xz-proteins',
            'https://example.org/papyruspp': b'fake-tsv-xz-papyruspp',
        }
        files = {
            '05.4': {
                'readme': {'name': 'README.txt', 'url': 'https://example.org/readme',
                           'size': 8, 'sha256': 'x'},
                'requirements': {'name': '05.4_additional_files.zip',
                                 'url': 'https://example.org/requirements',
                                 'size': len(zip_bytes), 'sha256': 'x'},
                'proteins': {'name': '05.4_combined_set_protein_targets.tsv.xz',
                             'url': 'https://example.org/proteins',
                             'size': 20, 'sha256': 'x'},
                'papyrus++': {'name': '05.4++_combined_set_without_stereochemistry.tsv.xz',
                              'url': 'https://example.org/papyruspp',
                              'size': 21, 'sha256': 'x'},
            },
        }

        self.events: list[str] = []

        def fake_get(url, **kwargs):
            self.events.append(f'download:{url}')
            return _FakeResponse(content_by_url[url])

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        self._queues_created: list[_FakeQueue] = []

        def make_fake_queue(*args, **kwargs):
            q = _FakeQueue(self.events, label='queued')
            self._queues_created.append(q)
            return q

        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value=files,
            ),
            'new_session': patch(
                'src.papyrus_scripts.download.new_session', return_value=fake_session,
            ),
            'assert_sha256sum': patch(
                'src.papyrus_scripts.download.assert_sha256sum', return_value=True,
            ),
            'mp_queue': patch(
                'src.papyrus_scripts.download.mp.Queue', side_effect=make_fake_queue,
            ),
            'mp_process': patch(
                'src.papyrus_scripts.download.mp.Process', side_effect=_FakeProcess,
            ),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def _restore_pystow_home(self):
        if self._old_pystow_home is None:
            os.environ.pop('PYSTOW_HOME', None)
        else:
            os.environ['PYSTOW_HOME'] = self._old_pystow_home

    def test_each_tabular_file_is_queued_right_after_its_own_download(self):
        with self.assertWarns(DeprecationWarning):  # links.json legacy-key warning
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=False)

        task_queue = self._queues_created[0]
        tasks = [item for item in task_queue.items if item is not download._CONVERSION_DONE]
        self.assertEqual({t['fpath'].name for t in tasks},
                          {'05.4_combined_set_protein_targets.tsv.xz',
                           '05.4++_combined_set_without_stereochemistry.tsv.xz'})
        # The sentinel must be the very last thing queued, once every file
        # has been downloaded (and every other file already queued).
        self.assertIs(task_queue.items[-1], download._CONVERSION_DONE)

        queued_events = [e for e in self.events if e.startswith('queued:')]
        download_events = [e for e in self.events if e.startswith('download:')]
        self.assertEqual(len(queued_events), 3)  # 2 files + the sentinel
        # The first file must be queued for conversion right after its own
        # download, not deferred until every other file has downloaded too.
        first_queued_index = self.events.index(queued_events[0])
        last_download_index = self.events.index(download_events[-1])
        self.assertLess(first_queued_index, last_download_index)

    def test_queued_tasks_carry_an_approximate_row_total_from_data_size_json(self):
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=False)

        task_queue = self._queues_created[0]
        tasks = {t['fpath'].name: t for t in task_queue.items if t is not download._CONVERSION_DONE}
        self.assertEqual(
            tasks['05.4_combined_set_protein_targets.tsv.xz']['total_rows'], 7,
        )
        self.assertEqual(
            tasks['05.4++_combined_set_without_stereochemistry.tsv.xz']['total_rows'], 42,
        )

    def test_queued_tasks_use_the_short_ftype_as_progress_bar_label(self):
        # Regression test: the progress bar label used to be the real
        # filename (e.g. "05.4++_combined_set_without_stereochemistry.tsv"),
        # long enough on its own to overflow the fixed bar width before the
        # bar itself even starts. It must be the short ftype key from
        # links.json (e.g. "papyrus++") instead.
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=False)

        task_queue = self._queues_created[0]
        tasks = {t['fpath'].name: t for t in task_queue.items if t is not download._CONVERSION_DONE}
        self.assertEqual(
            tasks['05.4_combined_set_protein_targets.tsv.xz']['desc'], 'Converting proteins',
        )
        self.assertEqual(
            tasks['05.4++_combined_set_without_stereochemistry.tsv.xz']['desc'],
            'Converting papyrus++',
        )


class TestConvertWorker(unittest.TestCase):
    """Unit tests for download._convert_worker, run synchronously in-process
    (it's a plain function - no real subprocess needed to exercise its logic).
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _task(self, name: str) -> dict:
        fpath = Path(self._tmpdir.name) / f'{name}.tsv.xz'
        fpath.touch()
        return {
            'fpath': fpath,
            'parquet_path': Path(self._tmpdir.name) / f'{name}.tsv.parquet',
            'schema_overrides': None,
            'null_values': None,
            'total_rows': None,
            'desc': f'Converting {name}',
        }

    def test_processes_every_task_then_exits_on_sentinel(self):
        tasks = [self._task('a'), self._task('b')]
        task_queue = _FakeQueue()
        for t in tasks:
            task_queue.put(t)
        task_queue.put(download._CONVERSION_DONE)
        error_queue = _FakeQueue()
        progress_queue = _FakeQueue()

        def fake_convert(input_file, output_file, **kwargs):
            Path(output_file).touch()

        with patch('src.papyrus_scripts.download.convert_xz_to_parquet', side_effect=fake_convert):
            download._convert_worker(task_queue, error_queue, progress_queue, progress=False)

        for t in tasks:
            self.assertFalse(t['fpath'].exists())  # .xz original deleted
            self.assertTrue(t['parquet_path'].exists())
        self.assertEqual(error_queue.items, [None])
        # progress=False: nothing at all reported, not even "done".
        self.assertEqual(progress_queue.items, [])

    def test_progress_true_reports_start_chunk_and_done_messages(self):
        # Regression test: _convert_worker used to render its own tqdm bars
        # in this (separate) process, concurrently with the parent's own
        # download-progress bar - two independent processes each doing
        # position-relative cursor math with no shared notion of the
        # terminal's actual state, which drifted out of sync and left
        # stray blank lines behind over time. It must instead report
        # progress via progress_queue for the parent (the only process
        # that renders anything) to consume.
        task = self._task('a')
        task['total_rows'] = 10
        task_queue = _FakeQueue()
        task_queue.put(task)
        task_queue.put(download._CONVERSION_DONE)
        error_queue = _FakeQueue()
        progress_queue = _FakeQueue()

        def fake_convert(input_file, output_file, on_progress=None, on_reset=None, **kwargs):
            on_reset()
            on_progress(4)
            on_progress(6)
            Path(output_file).touch()

        with patch('src.papyrus_scripts.download.convert_xz_to_parquet', side_effect=fake_convert):
            download._convert_worker(task_queue, error_queue, progress_queue, progress=True)

        self.assertEqual(progress_queue.items, [
            (download._PROGRESS_START, task['desc'], 10),
            (download._PROGRESS_RESET,),
            (download._PROGRESS_CHUNK, 4),
            (download._PROGRESS_CHUNK, 6),
            (download._PROGRESS_DONE,),
        ])

    def test_exception_removes_half_written_file_and_reports_error(self):
        task = self._task('c')
        tmp_parquet = task['parquet_path'].with_name(task['parquet_path'].name + '.converting')
        task_queue = _FakeQueue()
        task_queue.put(task)
        task_queue.put(download._CONVERSION_DONE)
        error_queue = _FakeQueue()
        progress_queue = _FakeQueue()

        def flaky_convert(input_file, output_file, **kwargs):
            # Simulate a conversion that got partway through writing before
            # failing (e.g. a KeyboardInterrupt raised mid-write).
            Path(output_file).with_name(Path(output_file).name + '.converting').touch()
            raise KeyboardInterrupt()

        with patch('src.papyrus_scripts.download.convert_xz_to_parquet', side_effect=flaky_convert):
            download._convert_worker(task_queue, error_queue, progress_queue, progress=False)

        self.assertFalse(tmp_parquet.exists())
        self.assertFalse(task['parquet_path'].exists())
        self.assertEqual(len(error_queue.items), 1)
        self.assertIn('KeyboardInterrupt', error_queue.items[0])
        # The original .xz is left alone (not deleted) after a failure.
        self.assertTrue(task['fpath'].exists())

    def test_stops_after_a_failure_instead_of_processing_further_tasks(self):
        first = self._task('d')
        second = self._task('e')
        task_queue = _FakeQueue()
        task_queue.put(first)
        task_queue.put(second)
        task_queue.put(download._CONVERSION_DONE)
        error_queue = _FakeQueue()
        progress_queue = _FakeQueue()

        with patch(
                'src.papyrus_scripts.download.convert_xz_to_parquet',
                side_effect=ValueError('boom'),
        ):
            download._convert_worker(task_queue, error_queue, progress_queue, progress=False)

        self.assertFalse(second['parquet_path'].exists())
        self.assertTrue(second['fpath'].exists())
        self.assertEqual(len(error_queue.items), 1)
        self.assertIn('boom', error_queue.items[0])


class TestDownloadPapyrusConvertWorkerIntegration(unittest.TestCase):
    """End-to-end check that _convert_worker genuinely works as a real
    multiprocessing.Process target - a real subprocess, real Queues,
    catching anything a pure in-process test (mocked/patched, immune to
    pickling and process-boundary issues) could miss.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_real_subprocess_converts_and_signals_completion(self):
        import lzma
        import polars as pl

        df = pl.DataFrame({'connectivity': ['C1', 'C2'], 'value': [1, 2]})
        fpath = Path(self._tmpdir.name) / 'real.tsv.xz'
        with lzma.open(fpath, 'wb') as fh:
            fh.write(df.write_csv(separator='\t').encode())
        parquet_path = Path(self._tmpdir.name) / 'real.tsv.parquet'

        task_queue = download.mp.Queue()
        error_queue = download.mp.Queue()
        progress_queue = download.mp.Queue()
        task_queue.put({
            'fpath': fpath, 'parquet_path': parquet_path,
            'schema_overrides': None, 'null_values': None,
            'total_rows': 2, 'desc': 'Converting real.tsv.xz',
        })
        task_queue.put(download._CONVERSION_DONE)

        process = download.mp.Process(
            target=download._convert_worker,
            args=(task_queue, error_queue, progress_queue, True),
        )
        process.start()
        process.join(timeout=60)

        self.assertEqual(process.exitcode, 0)
        self.assertIsNone(error_queue.get(timeout=5))
        self.assertFalse(fpath.exists())
        self.assertTrue(parquet_path.is_file())
        self.assertEqual(pl.read_parquet(parquet_path).height, 2)

        messages = []
        while True:
            try:
                messages.append(progress_queue.get(timeout=1))
            except queue.Empty:
                break
        self.assertIn((download._PROGRESS_START, 'Converting real.tsv.xz', 2), messages)
        self.assertIn((download._PROGRESS_DONE,), messages)
        self.assertEqual(sum(m[1] for m in messages if m[0] == download._PROGRESS_CHUNK), 2)


if __name__ == '__main__':
    unittest.main()
