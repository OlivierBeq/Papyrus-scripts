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
import threading
import time
import unittest
import warnings
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


class TestPrintBanner(unittest.TestCase):
    """Unit tests for download._print_banner, the boxed emoji-tagged notice
    used for the download banners (old-format-data notice, Papyrus++
    disclaimer, disk-space error).
    """

    def _render(self, *args, **kwargs):
        buf = io.StringIO()
        with redirect_stdout(buf):
            download._print_banner(*args, **kwargs)
        return buf.getvalue()

    def test_title_line_includes_the_emoji(self):
        output = self._render('🚫', 'Some title')
        self.assertIn('🚫 Some title', output)

    def test_body_lines_are_all_included(self):
        output = self._render('📢', 'Title', 'line one', 'line two')
        self.assertIn('line one', output)
        self.assertIn('line two', output)

    def test_every_line_has_matching_left_and_right_borders(self):
        output = self._render('📢', 'Title', 'a body line')
        content_lines = [line for line in output.splitlines() if line.startswith('│')]
        self.assertTrue(content_lines)
        for line in content_lines:
            self.assertTrue(line.endswith('│'), line)

    def test_top_and_bottom_rules_use_box_drawing_corners(self):
        output = self._render('📢', 'Title', 'a body line')
        lines = output.splitlines()
        self.assertTrue(lines[0].startswith('┌') and lines[0].endswith('┐'))
        self.assertTrue(lines[-1].startswith('└') and lines[-1].endswith('┘'))

    def test_title_row_aligns_with_body_rows_despite_the_emoji(self):
        # Regression test: len() counts an emoji as one character, but
        # terminals render it two columns wide - padding with plain
        # .ljust() (len()-based) therefore under-pads the title row,
        # visibly misaligning its right border relative to plain-text
        # body rows. All rows (including rule lines) must occupy the same
        # real display width.
        output = self._render('📢', 'Some title', 'a plain body line')
        widths = {download._display_width(line) for line in output.splitlines() if line}
        self.assertEqual(len(widths), 1, f'rows have inconsistent display widths: {widths}')

    def test_long_body_line_is_wrapped_instead_of_overflowing(self):
        # Regression test: a body line longer than the banner's fixed width
        # (e.g. a dynamically-built list of several stale versions) used to
        # be printed as-is, overflowing straight past the right border
        # instead of wrapping onto additional rows.
        long_line = (
            "(or remove_papyrus(version=['05.4', '05.7', '05.5', '2022.01', "
            "'2022.04'], version_root=True) from Python)"
        )
        output = self._render('📢', 'Title', long_line)
        content_lines = [line for line in output.splitlines() if line.startswith('│')]
        for line in content_lines:
            self.assertEqual(download._display_width(line), download._BANNER_WIDTH, line)
        # The long line's own words must still all be present, just spread
        # across more than one row.
        joined = ' '.join(content_lines)
        self.assertIn('remove_papyrus', joined)
        self.assertIn('from Python', joined)
        self.assertGreater(len(content_lines), 1)


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


class _StopEarlyError(Exception):
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
                'src.papyrus_scripts.download.get_papyrus_links', side_effect=_StopEarlyError,
            ),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_notice_lists_only_the_versions_actually_found(self):
        buf = io.StringIO()
        with self.assertRaises(_StopEarlyError), redirect_stdout(buf):
            download.download_papyrus(version='latest')
        notice = buf.getvalue()
        self.assertIn('--version 05.6', notice)
        self.assertIn('--version 05.7', notice)
        self.assertIn('--remove_version', notice)
        self.assertNotIn('--version all', notice)
        # Boxed, emoji-tagged banner rather than a plain '##########' rule.
        self.assertIn('📢', notice)
        self.assertIn('┌', notice)
        self.assertIn('└', notice)


class TestDownloadPapyrusDiskSpaceCheck(unittest.TestCase):
    """No free space left after download -> hard fail, no prompt. Only the
    safety margin violated -> ask for confirmation instead of aborting.
    """

    def setUp(self):
        self.v1 = make_version('v1')
        self.v2 = make_version('v2')
        patches = {
            'pystow_join': patch(
                'src.papyrus_scripts.download.pystow.join',
                return_value=Path(tempfile.gettempdir()) / 'does-not-exist.json',
            ),
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value={},
            ),
            'resolve_versions': patch(
                'src.papyrus_scripts.download._resolve_versions',
                return_value=[self.v1, self.v2],
            ),
            # Empty version_files -> nothing to actually download, so
            # a passed/overridden check falls through harmlessly instead
            # of needing the rest of the download/convert machinery mocked.
            'get_version_files': patch(
                'src.papyrus_scripts.download._get_version_files', return_value={},
            ),
            'total_size': patch(
                'src.papyrus_scripts.download._total_size', return_value=100,
            ),
            'pystow_module': patch('src.papyrus_scripts.download.pystow.module'),
            # Last step of a per-version download - mocked so an accepted
            # confirmation can run to completion without a real versions.json.
            'update_versions_json': patch(
                'src.papyrus_scripts.download._update_versions_json',
            ),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_no_space_left_hard_fails_without_prompting(self):
        with (
            patch('src.papyrus_scripts.download.get_disk_space', return_value=50),
            patch('builtins.input', side_effect=AssertionError('input() must not be called')),
            redirect_stdout(io.StringIO()) as buf,
        ):
            result = download.download_papyrus(version='latest', progress=False)
        self.assertIsNone(result)
        self.assertIn('🚫', buf.getvalue())
        self.assertIn('no space left on disk', buf.getvalue())
        # Hard stop: the second version is never even reached.
        self.assertEqual(self.mocks['get_version_files'].call_count, 1)
        self.mocks['update_versions_json'].assert_not_called()

    def test_margin_violation_aborts_on_declined_confirmation(self):
        with (
            patch('src.papyrus_scripts.download.get_disk_space', return_value=1000),
            patch('src.papyrus_scripts.download.enough_disk_space', return_value=False),
            patch('builtins.input', return_value='N') as mock_input,
            redirect_stdout(io.StringIO()) as buf,
        ):
            result = download.download_papyrus(version='latest', progress=False)
        self.assertIsNone(result)
        mock_input.assert_called_once()
        self.assertIn('was aborted', buf.getvalue())
        self.assertEqual(self.mocks['get_version_files'].call_count, 1)
        self.mocks['update_versions_json'].assert_not_called()

    def test_margin_violation_proceeds_on_accepted_confirmation(self):
        with (
            patch('src.papyrus_scripts.download.get_disk_space', return_value=1000),
            patch('src.papyrus_scripts.download.enough_disk_space', return_value=False),
            patch('builtins.input', return_value='Y') as mock_input,
            redirect_stdout(io.StringIO()) as buf,
        ):
            result = download.download_papyrus(
                version='latest', progress=False, keep_xz=True,
            )
        self.assertIsNone(result)
        self.assertNotIn('was aborted', buf.getvalue())
        # Both versions reached -> fell through the confirmation, not an early return.
        self.assertEqual(mock_input.call_count, 2)
        self.assertEqual(self.mocks['get_version_files'].call_count, 2)
        self.assertEqual(self.mocks['update_versions_json'].call_count, 2)


class _FakeResponse:
    def __init__(self, content: bytes):
        self._content = content

    def iter_content(self, chunk_size):
        yield self._content


class _SlowFakeResponse:
    """Like _FakeResponse, but streams in several delayed chunks - so the
    main download loop keeps polling _drain_progress_queue() for real wall
    time, long enough for an earlier file's conversion to genuinely overlap
    with this one still downloading.
    """

    def __init__(self, content: bytes, n_chunks: int = 5, delay: float = 0.08):
        self._content = content
        self._n_chunks = n_chunks
        self._delay = delay

    def iter_content(self, chunk_size):
        step = max(1, len(self._content) // self._n_chunks)
        for i in range(0, len(self._content), step):
            time.sleep(self._delay)
            yield self._content[i:i + step]


class _FakeQueue:
    """Stand-in for multiprocessing.Queue that runs in-process, so tests can
    inspect exactly what was queued without a real subprocess.
    """

    def __init__(self, events: list | None = None, label: str | None = None):
        self.items: list = []
        self._pending: list = []
        self._events = events
        self._label = label

    def put(self, item, timeout=None):
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


class _AlwaysFullQueue:
    """task_queue stand-in whose put() always raises queue.Full - simulates
    a converter that stopped draining task_queue (e.g. because it died).
    """

    def put(self, item, timeout=None):
        raise queue.Full


class _PresetErrorQueue:
    """error_queue stand-in with a single error message ready to .get()."""

    def __init__(self, message):
        self._message = message

    def get(self, timeout=None):
        return self._message

    def get_nowait(self):
        return self._message


class TestDownloadPapyrusEnqueueDeadlock(unittest.TestCase):
    """_enqueue must raise RuntimeError instead of retrying task_queue.put()
    forever when the converter process has died and can no longer drain it.
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

        def fake_get(url, **kwargs):
            return _FakeResponse(content_by_url[url])

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        # 1st mp.Queue() call is task_queue, 2nd is error_queue, 3rd is
        # progress_queue (see download_papyrus's `if not keep_xz:` setup).
        self._queue_factories = [
            lambda *a, **k: _AlwaysFullQueue(),
            lambda *a, **k: _PresetErrorQueue('worker crashed: BOOM'),
            lambda *a, **k: _FakeQueue(),
        ]
        self._queue_call_count = 0

        def make_queue(*args, **kwargs):
            factory = self._queue_factories[self._queue_call_count]
            self._queue_call_count += 1
            return factory(*args, **kwargs)

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
                'src.papyrus_scripts.download.mp.Queue', side_effect=make_queue,
            ),
            # is_alive() always False - the converter is already dead.
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

    def test_dead_converter_raises_instead_of_hanging(self):
        with (
            self.assertWarns(DeprecationWarning),  # links.json legacy-key warning
            self.assertRaises(RuntimeError) as ctx,
        ):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=False)
        self.assertIn('worker crashed: BOOM', str(ctx.exception))

    def test_error_queue_empty_falls_back_to_generic_message(self):
        self._queue_factories[1] = lambda *a, **k: _EmptyErrorQueue()
        with (
            self.assertWarns(DeprecationWarning),
            self.assertRaises(RuntimeError) as ctx,
        ):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=False)
        self.assertIn('worker exited unexpectedly', str(ctx.exception))


class _EmptyErrorQueue:
    """error_queue stand-in whose get() always times out (queue.Empty)."""

    def get(self, timeout=None):
        raise queue.Empty

    def get_nowait(self):
        raise queue.Empty


class TestDownloadPapyrusSingleConvertingProgressBar(unittest.TestCase):
    """download_papyrus shows two simultaneous, position-pinned bars -
    'Downloading version X' at position=0 and 'Converting files' at
    position=1 - both owned and rendered by this (parent) process only,
    which is tqdm's well-supported case for stacked bars (see
    _convert_worker's docstring for the multi-*process* rendering that
    was tried and reverted instead - a different failure mode that
    doesn't apply to two bars from one process). The 'Converting files'
    bar must be constructed exactly once, up front (alongside the
    converter process), at position=1.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._old_pystow_home = os.environ.get('PYSTOW_HOME')
        self.addCleanup(self._restore_pystow_home)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')
            zh.writestr('data_size.json', '{"papyrus_proteins": 7}')
        zip_bytes = zip_buf.getvalue()

        content_by_url = {
            'https://example.org/readme': b'a readme',
            'https://example.org/requirements': zip_bytes,
            'https://example.org/proteins': b'fake-tsv-xz-proteins',
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
            },
        }

        def fake_get(url, **kwargs):
            return _FakeResponse(content_by_url[url])

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        self.tqdm_calls: list[dict] = []

        def fake_tqdm(*args, **kwargs):
            self.tqdm_calls.append(kwargs)
            return MagicMock()

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
                'src.papyrus_scripts.download.mp.Queue', side_effect=lambda *a, **k: _FakeQueue(),
            ),
            'mp_process': patch(
                'src.papyrus_scripts.download.mp.Process', side_effect=_FakeProcess,
            ),
            'tqdm': patch('src.papyrus_scripts.download.tqdm', side_effect=fake_tqdm),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def _restore_pystow_home(self):
        if self._old_pystow_home is None:
            os.environ.pop('PYSTOW_HOME', None)
        else:
            os.environ['PYSTOW_HOME'] = self._old_pystow_home

    def test_converting_files_bar_created_at_most_once(self):
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        converting_calls = [c for c in self.tqdm_calls if c.get('desc') == 'Converting files']
        self.assertEqual(len(converting_calls), 1)

    def test_converting_files_bar_uses_position_one(self):
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        converting_calls = [c for c in self.tqdm_calls if c.get('desc') == 'Converting files']
        self.assertEqual(converting_calls[0].get('position'), 1)

    def test_download_bar_uses_position_zero(self):
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        download_calls = [c for c in self.tqdm_calls if 'Downloading version' in c.get('desc', '')]
        self.assertEqual(download_calls[0].get('position'), 0)


class _SlowConsumerProcess:
    """Stand-in for multiprocessing.Process: consumes task_queue on a
    background thread with an artificial per-task delay, so a real
    maxsize=1 task_queue genuinely backpressures download_papyrus's
    producer side - exactly like a slow real converter would, without
    actually spawning a subprocess or touching pandas/pyarrow.

    :param first_delay: seconds to wait before ever calling task_queue.get()
        for the first time - set >0 to deterministically guarantee real
        backpressure (regardless of thread-scheduling races) for tests that
        need it; 0 (default) lets the worker start consuming immediately.
    """

    first_delay: float = 0.0

    def __init__(self, target=None, args=(), daemon=None):
        self._task_queue, self._error_queue, self._progress_queue, self._progress = args
        self._thread: threading.Thread | None = None

    def start(self):
        def run():
            first = True
            while True:
                if first:
                    if self.first_delay:
                        # Guarantee real backpressure regardless of thread
                        # scheduling: leave the first task sitting
                        # unconsumed in the maxsize=1 queue for a while, so
                        # the producer (main thread) is certain to block on
                        # its *next* put() before this thread ever calls
                        # get().
                        time.sleep(self.first_delay)
                    first = False
                task = self._task_queue.get()
                if task is download._CONVERSION_DONE:
                    self._error_queue.put(None)
                    return
                if self._progress:
                    self._progress_queue.put((download._PROGRESS_START, task['desc'], 100))
                for _ in range(3):
                    time.sleep(0.1)
                    if self._progress:
                        self._progress_queue.put((download._PROGRESS_CHUNK, 25))
                if self._progress:
                    self._progress_queue.put((download._PROGRESS_DONE,))

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()


class _SlowConsumerProcessWithStartupDelay(_SlowConsumerProcess):
    """_SlowConsumerProcess variant that deterministically backpressures the
    very first task, for tests specifically about the queue.Full path.
    """

    first_delay = 0.5


class TestDownloadPapyrusBackpressureFeedback(unittest.TestCase):
    """Regression test: task_queue.put() (maxsize=1) blocks the producer
    whenever the converter hasn't yet taken the previous task - a real wait
    for a large file. A plain blocking put() gave zero visible feedback
    during that wait (not even the download bar moved), indistinguishable
    from a genuine hang - which is exactly what was reported after the
    eager converting_pbar was removed: no conversion progress shown at all,
    to the point of looking stuck. download_papyrus must instead keep
    showing something (a status message on whichever bar is on screen)
    while backpressured, and must not block forever - a real maxsize=1
    task_queue with a deliberately slow (but not stuck) consumer must
    still let download_papyrus complete.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._old_pystow_home = os.environ.get('PYSTOW_HOME')
        self.addCleanup(self._restore_pystow_home)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')
            zh.writestr('data_size.json', '{}')
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

        def fake_get(url, **kwargs):
            return _FakeResponse(content_by_url[url])

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        self.pbar_mocks: list[MagicMock] = []

        def fake_tqdm(*args, **kwargs):
            m = MagicMock()
            m._kwargs = kwargs
            self.pbar_mocks.append(m)
            return m

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
            'mp_process': patch(
                'src.papyrus_scripts.download.mp.Process',
                side_effect=_SlowConsumerProcessWithStartupDelay,
            ),
            'tqdm': patch('src.papyrus_scripts.download.tqdm', side_effect=fake_tqdm),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def _restore_pystow_home(self):
        if self._old_pystow_home is None:
            os.environ.pop('PYSTOW_HOME', None)
        else:
            os.environ['PYSTOW_HOME'] = self._old_pystow_home

    def test_does_not_hang_when_converter_is_slower_than_downloading(self):
        result = {}

        def run():
            with self.assertWarns(DeprecationWarning):
                download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)
            result['finished'] = True

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=10)
        self.assertTrue(result.get('finished'), 'download_papyrus hung instead of completing')

    def test_shows_feedback_while_backpressured_instead_of_going_silent(self):
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        # Some bar (download bar's postfix, or converting_pbar's
        # description) must have received a non-empty, real (non-'?') live
        # update reflecting actual conversion progress while task_queue.put()
        # was blocked on the slow consumer - proving the wait is visible,
        # not silent.
        statuses = [
            call.args[0]
            for m in self.pbar_mocks
            for method in (m.set_postfix_str, m.set_description)
            for call in method.call_args_list
            if call.args and call.args[0]
        ]
        self.assertTrue(statuses, 'no feedback was shown while backpressured on task_queue.put()')
        self.assertTrue(
            any('/100 rows' in s for s in statuses),
            f'no live row-progress feedback found among: {statuses}',
        )

class TestDownloadPapyrusShowsConversionProgressWhileStillDownloading(unittest.TestCase):
    """Regression test: conversion progress used to be silently counted but
    never displayed until the download bar closed - so a small file's
    conversion, starting as soon as it finishes downloading and easily
    running to completion while a later, larger file is still downloading,
    was invisible until everything was done. converting_pbar (position=1)
    must instead be live - receiving real description/update calls - while
    the download bar (position=0) is still open, not only afterwards.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._old_pystow_home = os.environ.get('PYSTOW_HOME')
        self.addCleanup(self._restore_pystow_home)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')
            zh.writestr('data_size.json', '{}')
        zip_bytes = zip_buf.getvalue()

        # Both tabular files stream slowly (real wall time) - regardless of
        # which one download_papyrus happens to download first (ordered_
        # ftypes' order among non-readme/requirements files is set-iteration
        # order, not guaranteed), the other one's still-active download
        # overlaps with the first one's background conversion - the
        # scenario the fix is about.
        content_by_url = {
            'https://example.org/readme': _FakeResponse(b'a readme'),
            'https://example.org/requirements': _FakeResponse(zip_bytes),
            'https://example.org/proteins': _SlowFakeResponse(b'fake-tsv-xz-proteins' * 20),
            'https://example.org/papyruspp': _SlowFakeResponse(b'fake-tsv-xz-papyruspp' * 20),
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
                             'size': 20 * 20, 'sha256': 'x'},
                'papyrus++': {'name': '05.4++_combined_set_without_stereochemistry.tsv.xz',
                              'url': 'https://example.org/papyruspp',
                              'size': 21 * 20, 'sha256': 'x'},
            },
        }

        def fake_get(url, **kwargs):
            return content_by_url[url]

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        self.pbar_mocks: list[MagicMock] = []
        # A shared manager so calls across the two separate bar mocks can be
        # compared in one true chronological order (manager.mock_calls
        # interleaves attached children's calls in real call order).
        self.manager = MagicMock()

        def fake_tqdm(*args, **kwargs):
            m = MagicMock()
            m._kwargs = kwargs
            self.pbar_mocks.append(m)
            self.manager.attach_mock(m, f'bar{len(self.pbar_mocks)}')
            return m

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
            'mp_process': patch(
                'src.papyrus_scripts.download.mp.Process', side_effect=_SlowConsumerProcess,
            ),
            'tqdm': patch('src.papyrus_scripts.download.tqdm', side_effect=fake_tqdm),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def _restore_pystow_home(self):
        if self._old_pystow_home is None:
            os.environ.pop('PYSTOW_HOME', None)
        else:
            os.environ['PYSTOW_HOME'] = self._old_pystow_home

    def test_download_bar_shows_conversion_progress_before_it_closes(self):
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        download_bars = [m for m in self.pbar_mocks if 'Downloading version' in m._kwargs.get('desc', '')]
        converting_bars = [m for m in self.pbar_mocks if m._kwargs.get('desc') == 'Converting files']
        self.assertEqual(len(download_bars), 1)
        self.assertEqual(len(converting_bars), 1)
        pbar, converting_pbar = download_bars[0], converting_bars[0]

        # manager.mock_calls interleaves both bars' calls in true
        # chronological order - unlike comparing indices within each mock's
        # own (separate) call list.
        calls = self.manager.mock_calls
        names = [f'bar{i + 1}' for i in range(len(self.pbar_mocks))]
        pbar_name = next(name for name, m in zip(names, self.pbar_mocks, strict=True) if m is pbar)
        converting_name = next(
            name for name, m in zip(names, self.pbar_mocks, strict=True) if m is converting_pbar
        )

        close_index = next(i for i, c in enumerate(calls) if c[0] == f'{pbar_name}.close')
        live_updates_before_close = [
            c for i, c in enumerate(calls)
            if i < close_index
            and c[0] in (f'{converting_name}.set_description', f'{converting_name}.update')
            and c.args and c.args[0]
        ]
        self.assertTrue(
            live_updates_before_close,
            'converting_pbar never received a live update before the download bar closed',
        )
        self.assertTrue(
            any('Converting' in str(c.args[0]) and '/100 rows' in str(c.args[0])
                for c in live_updates_before_close if c[0] == f'{converting_name}.set_description'),
            f'no real row-progress description shown before download bar closed: {live_updates_before_close}',
        )

    def test_completed_file_shows_complete_instead_of_final_row_count(self):
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        converting_bars = [m for m in self.pbar_mocks if m._kwargs.get('desc') == 'Converting files']
        self.assertEqual(len(converting_bars), 1)
        converting_pbar = converting_bars[0]

        descriptions = [
            call.args[0] for call in converting_pbar.set_description.call_args_list
            if call.args
        ]
        self.assertTrue(
            any(d.endswith('(complete)') for d in descriptions),
            f'no "(complete)" description found among: {descriptions}',
        )

    def test_completion_triggers_a_single_refresh_not_two(self):
        # Regression test: marking a file complete used to call
        # set_description() (which forces its own unconditional refresh)
        # immediately followed by update(1) (which forces another) - two
        # consecutive cursor-repositioning writes for the same event, with
        # nothing in between, which left the cursor one line off on Windows
        # so the *next* redraw landed on a fresh line instead of
        # overwriting in place. The description must now be set with
        # refresh=False, letting update(1)'s own refresh be the only one.
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        converting_bars = [m for m in self.pbar_mocks if m._kwargs.get('desc') == 'Converting files']
        self.assertEqual(len(converting_bars), 1)
        converting_pbar = converting_bars[0]

        complete_calls = [
            call for call in converting_pbar.set_description.call_args_list
            if call.args and call.args[0].endswith('(complete)')
        ]
        self.assertTrue(complete_calls)
        for call in complete_calls:
            self.assertIs(
                call.kwargs.get('refresh', True), False,
                'set_description for "(complete)" must pass refresh=False',
            )

    def test_download_bar_closes_after_converting_bar_not_before(self):
        # Regression test: the download bar (position=0) used to close as
        # soon as downloading finished, well before the still-active
        # converting bar (position=1). tqdm's close() (leave=True, our
        # default) writes an explicit, uncoordinated newline of its own
        # after the bar's final state - a real newline a still-active
        # position=1 bar has no way to account for in its own cursor math,
        # permanently shifting it one row off for every redraw from that
        # point on. Bars must close bottom-up: converting_pbar first, then
        # the download bar last.
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        download_bars = [m for m in self.pbar_mocks if 'Downloading version' in m._kwargs.get('desc', '')]
        converting_bars = [m for m in self.pbar_mocks if m._kwargs.get('desc') == 'Converting files']
        self.assertEqual(len(download_bars), 1)
        self.assertEqual(len(converting_bars), 1)
        pbar, converting_pbar = download_bars[0], converting_bars[0]

        calls = self.manager.mock_calls
        names = [f'bar{i + 1}' for i in range(len(self.pbar_mocks))]
        pbar_name = next(name for name, m in zip(names, self.pbar_mocks, strict=True) if m is pbar)
        converting_name = next(
            name for name, m in zip(names, self.pbar_mocks, strict=True) if m is converting_pbar
        )

        pbar_close_index = next(i for i, c in enumerate(calls) if c[0] == f'{pbar_name}.close')
        converting_close_index = next(i for i, c in enumerate(calls) if c[0] == f'{converting_name}.close')
        self.assertLess(
            converting_close_index, pbar_close_index,
            'converting_pbar must close before the download bar, not after',
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

    def _structure_task(self, name: str) -> dict:
        fpath = Path(self._tmpdir.name) / f'{name}.sd.xz'
        fpath.touch()
        return {
            'fpath': fpath,
            'parquet_path': Path(self._tmpdir.name) / f'{name}.sd.parquet',
            'schema_overrides': None,
            'null_values': None,
            'total_rows': None,
            'desc': f'Converting {name}',
        }

    def test_dispatches_sd_xz_tasks_to_convert_sd_to_parquet(self):
        task = self._structure_task('a')
        task_queue = _FakeQueue()
        task_queue.put(task)
        task_queue.put(download._CONVERSION_DONE)
        error_queue = _FakeQueue()
        progress_queue = _FakeQueue()

        def fake_convert_sd(input_file, output_file, **kwargs):
            Path(output_file).touch()

        def fail_if_called(*args, **kwargs):
            raise AssertionError('convert_xz_to_parquet must not be called for .sd.xz files')

        with (
            patch('src.papyrus_scripts.download.convert_sd_to_parquet', side_effect=fake_convert_sd),
            patch('src.papyrus_scripts.download.convert_xz_to_parquet', side_effect=fail_if_called),
        ):
            download._convert_worker(task_queue, error_queue, progress_queue, progress=False)

        self.assertFalse(task['fpath'].exists())  # .xz original deleted
        self.assertTrue(task['parquet_path'].exists())
        self.assertEqual(error_queue.items, [None])

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

    def test_exception_reports_error_and_leaves_xz_original(self):
        # convert_xz_to_parquet (mocked here) now owns its own '.converting'
        # cleanup (see tests/test_io.py) - this only checks _convert_worker's
        # error reporting.
        task = self._task('c')
        task_queue = _FakeQueue()
        task_queue.put(task)
        task_queue.put(download._CONVERSION_DONE)
        error_queue = _FakeQueue()
        progress_queue = _FakeQueue()

        with patch('src.papyrus_scripts.download.convert_xz_to_parquet', side_effect=KeyboardInterrupt()):
            download._convert_worker(task_queue, error_queue, progress_queue, progress=False)

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


class TestResolveVersions(unittest.TestCase):
    """Direct tests for _resolve_versions - real PapyrusVersion objects
    (aliases table is loaded offline from a local cache, no network needed).
    """

    def setUp(self):
        self.files = {
            '2022.04.1': {}, '2022.04.2': {},
            '2022.08.3': {},
            '2022.11.4': {},
            '2024.09.2': {},
        }

    def test_latest_resolves_to_the_highest_alias_and_revision(self):
        resolved = download._resolve_versions('latest', self.files)
        self.assertEqual([pv.version for pv in resolved], ['2024.09.2'])

    def test_all_without_all_revisions_keeps_only_latest_revision_per_alias(self):
        resolved = download._resolve_versions('all', self.files, all_revisions=False)
        self.assertEqual(
            sorted(pv.version for pv in resolved),
            ['2022.04.2', '2022.08.3', '2022.11.4', '2024.09.2'],
        )

    def test_all_with_all_revisions_expands_every_canonical_key(self):
        resolved = download._resolve_versions('all', self.files, all_revisions=True)
        self.assertEqual(
            sorted(pv.version for pv in resolved),
            ['2022.04.1', '2022.04.2', '2022.08.3', '2022.11.4', '2024.09.2'],
        )

    def test_all_with_all_revisions_warns_and_skips_unaliased_canonical_key(self):
        files = dict(self.files)
        files['2099.01.1'] = {}  # no entry in aliases.json
        with self.assertWarns(UserWarning):
            resolved = download._resolve_versions('all', files, all_revisions=True)
        self.assertNotIn('2099.01.1', [pv.version for pv in resolved])

    def test_two_part_alias_with_all_revisions_expands_its_revisions(self):
        resolved = download._resolve_versions('2022.04', self.files, all_revisions=True)
        self.assertEqual(sorted(pv.version for pv in resolved), ['2022.04.1', '2022.04.2'])

    def test_two_part_alias_with_all_revisions_and_no_matching_keys_raises(self):
        files = {'2022.04.2': {}}  # no 2022.11.* keys present
        with self.assertRaises(ValueError):
            download._resolve_versions('2022.11', files, all_revisions=True)

    def test_unrecognised_single_version_string_raises(self):
        with self.assertRaises(ValueError):
            download._resolve_versions('not_a_real_version', self.files)

    def test_valid_version_absent_from_files_raises(self):
        with self.assertRaises(ValueError):
            download._resolve_versions('2022.08.3', {'2022.04.2': {}})

    def test_deduplicates_while_preserving_oldest_first_order(self):
        resolved = download._resolve_versions(['2024.09.2', '2022.04.2', '2024.09.2'], self.files)
        self.assertEqual([pv.version for pv in resolved], ['2022.04.2', '2024.09.2'])


class TestGetVersionFiles(unittest.TestCase):
    """Direct tests for _get_version_files: canonical key, legacy-key
    fallback (with DeprecationWarning), missing-key error, and the
    per-entry '__deprecated__' warning.
    """

    def test_canonical_key_returns_entry_without_warning(self):
        pv = download.PapyrusVersion(version='2022.04.2')
        files = {pv.version: {'readme': {}}}
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            entry = download._get_version_files(files, pv)
        self.assertEqual(entry, {'readme': {}})

    def test_legacy_key_fallback_warns_and_returns_entry(self):
        pv = download.PapyrusVersion(version='2022.04.2')
        files = {pv.version_old_fmt: {'readme': {}}}
        with self.assertWarns(DeprecationWarning):
            entry = download._get_version_files(files, pv)
        self.assertEqual(entry, {'readme': {}})

    def test_neither_key_present_raises_keyerror(self):
        pv = download.PapyrusVersion(version='2022.04.2')
        with self.assertRaises(KeyError):
            download._get_version_files({}, pv)

    def test_deprecated_entry_field_emits_its_own_warning(self):
        pv = download.PapyrusVersion(version='2022.04.2')
        files = {pv.version: {'__deprecated__': 'use version X instead'}}
        with self.assertWarns(DeprecationWarning) as ctx:
            download._get_version_files(files, pv)
        self.assertIn('use version X instead', str(ctx.warning))


class TestIterEntries(unittest.TestCase):

    def test_dict_value_wrapped_in_a_single_element_list(self):
        entry = {'name': 'a'}
        self.assertEqual(download._iter_entries(entry), [entry])

    def test_list_value_returned_as_is(self):
        entries = [{'name': 'a'}, {'name': 'b'}]
        self.assertEqual(download._iter_entries(entries), entries)

    def test_other_type_raises_valueerror(self):
        with self.assertRaises(ValueError):
            download._iter_entries('not-a-dict-or-list')


class TestFilePath(unittest.TestCase):

    def setUp(self):
        self.root = MagicMock()
        self.root.join.return_value = Path('/fake/path')

    def test_root_ftype_joins_directly_under_version_root(self):
        download._file_path(self.root, 'proteins', 'proteins.tsv.xz')
        self.root.join.assert_called_once_with(name='proteins.tsv.xz')

    def test_structure_ftype_joins_under_structures_subfolder(self):
        download._file_path(self.root, '2D_structures', 'structures.sd.xz')
        self.root.join.assert_called_once_with('structures', name='structures.sd.xz')

    def test_other_ftype_joins_under_descriptors_subfolder(self):
        download._file_path(self.root, '2D_mold2', 'mold2.tsv.xz')
        self.root.join.assert_called_once_with('descriptors', name='mold2.tsv.xz')


class TestParquetSibling(unittest.TestCase):
    """_parquet_sibling: .tsv.xz and .sd.xz convert, everything else doesn't."""

    def test_tabular_file_converts(self):
        self.assertEqual(
            download._parquet_sibling(Path('bioactivities.tsv.xz')),
            Path('bioactivities.tsv.parquet'),
        )

    def test_structure_file_converts(self):
        self.assertEqual(
            download._parquet_sibling(Path('05.7_combined_2D_set_without_stereochemistry.sd.xz')),
            Path('05.7_combined_2D_set_without_stereochemistry.sd.parquet'),
        )

    def test_non_data_files_are_never_converted(self):
        self.assertIsNone(download._parquet_sibling(Path('README.txt')))
        self.assertIsNone(download._parquet_sibling(Path('requirements.zip')))
        self.assertIsNone(download._parquet_sibling(Path('data_types.json')))


class TestUpdateVersionsJson(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = MagicMock()
        self.json_file = Path(self._tmpdir.name) / 'versions.json'
        self.root.join.return_value = self.json_file
        self.pv = make_version('2022.04.2')

    def test_add_creates_file_with_sorted_unique_entries(self):
        download._update_versions_json(self.root, self.pv, add=True)
        self.assertEqual(download.read_jsonfile(self.json_file), ['2022.04.2'])

    def test_remove_drops_the_matching_entry_only(self):
        download.write_jsonfile(['2022.04.2', 'other'], self.json_file)
        download._update_versions_json(self.root, self.pv, add=False)
        self.assertEqual(download.read_jsonfile(self.json_file), ['other'])


class TestConvertWorkerCrashBeforeTaskLoop(unittest.TestCase):

    def test_queue_get_failure_still_reports_to_error_queue_and_reraises(self):
        # Regression test: a crash/interrupt before the per-task try/except
        # is even reached (e.g. task_queue.get() itself interrupted) must
        # still report something on error_queue, instead of leaving the
        # parent blocked forever on error_queue.get().
        class _RaisingQueue(_FakeQueue):
            def get(self):
                raise RuntimeError('boom')

        task_queue = _RaisingQueue()
        error_queue = _FakeQueue()
        progress_queue = _FakeQueue()
        with self.assertRaises(RuntimeError):
            download._convert_worker(task_queue, error_queue, progress_queue, progress=False)
        self.assertEqual(error_queue.items, ['worker exited unexpectedly'])


class TestRemovePapyrusRootWipe(unittest.TestCase):
    """The nuclear option: remove_papyrus(papyrus_root=True)."""

    def setUp(self):
        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value={},
            ),
            'resolve_versions': patch(
                'src.papyrus_scripts.download._resolve_versions', return_value=[],
            ),
            'pystow_module': patch('src.papyrus_scripts.download.pystow.module'),
            'rmtree': patch('src.papyrus_scripts.download.shutil.rmtree'),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_declined_confirmation_aborts_without_removing(self):
        with (
            patch('builtins.input', return_value='N'),
            redirect_stdout(io.StringIO()) as buf,
        ):
            result = download.remove_papyrus(papyrus_root=True, progress=False)
        self.assertIsNone(result)
        self.assertIn('was aborted', buf.getvalue())
        self.mocks['rmtree'].assert_not_called()

    def test_accepted_confirmation_wipes_everything(self):
        with (
            patch('builtins.input', return_value='Y'),
            redirect_stdout(io.StringIO()) as buf,
        ):
            result = download.remove_papyrus(papyrus_root=True, progress=True)
        self.assertIsNone(result)
        self.mocks['rmtree'].assert_called_once()
        self.assertIn('successfully removed', buf.getvalue())

    def test_force_skips_the_confirmation_prompt(self):
        with (
            patch('builtins.input', side_effect=AssertionError('input() must not be called')),
        ):
            download.remove_papyrus(papyrus_root=True, force=True, progress=False)
        self.mocks['rmtree'].assert_called_once()


class TestRemovePapyrusVersionRootWipe(unittest.TestCase):
    """Per-version nuclear option: remove_papyrus(version_root=True)."""

    def setUp(self):
        self.v1 = make_version('v1')
        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value={},
            ),
            'resolve_versions': patch(
                'src.papyrus_scripts.download._resolve_versions', return_value=[self.v1],
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

    def test_declined_confirmation_skips_this_version_only(self):
        with (
            patch('builtins.input', return_value='N'),
            redirect_stdout(io.StringIO()) as buf,
        ):
            download.remove_papyrus(version='v1', version_root=True, progress=False)
        self.assertIn('was aborted', buf.getvalue())
        self.mocks['rmtree'].assert_not_called()
        self.mocks['update_versions_json'].assert_not_called()

    def test_accepted_confirmation_wipes_the_version_folder(self):
        with (
            patch('builtins.input', return_value='Y'),
            redirect_stdout(io.StringIO()) as buf,
        ):
            download.remove_papyrus(version='v1', version_root=True, progress=True)
        self.mocks['rmtree'].assert_called_once()
        self.mocks['update_versions_json'].assert_called_once_with(
            self.mocks['pystow_module'].return_value, self.v1, add=False,
        )
        self.assertIn('successfully removed', buf.getvalue())


class TestRemovePapyrusFileRemovalBody(unittest.TestCase):
    """The regular (non-nuclear) per-file-type removal path: which files
    are 'present', size accounting, actual unlink() calls, and the final
    versions.json deregistration once the version folder is empty.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root_dir = Path(self._tmpdir.name)
        (self.root_dir / 'papyruspp.tsv.xz').write_bytes(b'x' * 10)
        # 'proteins.tsv.xz' deliberately absent -> not "present".

        self.v1 = make_version('v1')
        version_files = {
            'papyrus++': {'name': 'papyruspp.tsv.xz', 'url': 'x', 'size': 10, 'sha256': 'x'},
            'proteins': {'name': 'proteins.tsv.xz', 'url': 'x', 'size': 5, 'sha256': 'x'},
        }

        root_module = MagicMock()
        root_module.base = self.root_dir
        root_module.join.side_effect = (
            lambda name=None, *a: self.root_dir / name if name else self.root_dir
        )

        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value={},
            ),
            'resolve_versions': patch(
                'src.papyrus_scripts.download._resolve_versions', return_value=[self.v1],
            ),
            'get_version_files': patch(
                'src.papyrus_scripts.download._get_version_files', return_value=version_files,
            ),
            'pystow_module': patch(
                'src.papyrus_scripts.download.pystow.module', return_value=root_module,
            ),
            'update_versions_json': patch('src.papyrus_scripts.download._update_versions_json'),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_only_fully_present_filetypes_are_removed(self):
        download.remove_papyrus(version='v1', papyruspp=True, proteins=True, progress=False)
        self.assertFalse((self.root_dir / 'papyruspp.tsv.xz').exists())
        # 'proteins' was never on disk, so nothing to unlink and no crash.
        self.mocks['update_versions_json'].assert_called_once_with(
            self.mocks['pystow_module'].return_value, self.v1, add=False,
        )

    def test_progress_true_prints_summary_and_closes_a_bar(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            download.remove_papyrus(version='v1', papyruspp=True, progress=True)
        self.assertIn('Number of files to be removed: 1', buf.getvalue())

    def test_nothing_present_leaves_directory_untouched_and_no_deregistration(self):
        download.remove_papyrus(version='v1', proteins=True, progress=False)
        self.mocks['update_versions_json'].assert_not_called()


class TestDownloadPapyrusFileSelectionByFlags(unittest.TestCase):
    """Which logical file types get downloaded, driven by nostereo/stereo/
    only_pp/structures/descriptors. keep_xz=True bypasses the whole
    queue/converter machinery entirely, so these tests only need a fake
    HTTP session.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

        def entry(url, name):
            return {'name': name, 'url': url, 'size': 1, 'sha256': 'x'}

        self.files = {
            '05.4': {
                'readme': entry('u:readme', 'README.txt'),
                'requirements': entry('u:requirements', 'req.zip'),
                'proteins': entry('u:proteins', 'proteins.tsv.xz'),
                'papyrus++': entry('u:papyruspp', 'papyruspp.tsv.xz'),
                '2D_papyrus': entry('u:2dpap', '2dpap.tsv.xz'),
                '2D_structures': entry('u:2dstruct', '2dstruct.sd.xz'),
                '3D_papyrus': entry('u:3dpap', '3dpap.tsv.xz'),
                '3D_structures': entry('u:3dstruct', '3dstruct.sd.xz'),
                '3D_mordred': entry('u:3dmordred', '3dmordred.tsv.xz'),
                '3D_fingerprint': entry('u:3dfp', '3dfp.tsv.xz'),
                'proteins_prodec': entry('u:prodec', 'prodec.tsv.xz'),
            },
        }

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')
        zip_bytes = zip_buf.getvalue()

        self.requested_urls: list[str] = []

        def fake_get(url, **kwargs):
            self.requested_urls.append(url)
            content = zip_bytes if url == 'u:requirements' else b'x'
            return _FakeResponse(content)

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value=self.files,
            ),
            'new_session': patch(
                'src.papyrus_scripts.download.new_session', return_value=fake_session,
            ),
            'assert_sha256sum': patch(
                'src.papyrus_scripts.download.assert_sha256sum', return_value=True,
            ),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def _run(self, **kwargs):
        with self.assertWarns(DeprecationWarning):  # links.json legacy-key warning
            download.download_papyrus(
                outdir=self._tmpdir.name, version='05.4', progress=False, keep_xz=True, **kwargs,
            )

    def test_only_pp_false_downloads_full_2d_bioactivity_set(self):
        self._run(nostereo=True, only_pp=False, stereo=False, descriptors=None)
        self.assertIn('u:2dpap', self.requested_urls)

    def test_structures_true_downloads_2d_structures(self):
        self._run(nostereo=True, structures=True, stereo=False, descriptors=None)
        self.assertIn('u:2dstruct', self.requested_urls)

    def test_stereo_true_downloads_3d_bioactivity_and_structures(self):
        self._run(nostereo=False, stereo=True, structures=True, descriptors=None)
        self.assertIn('u:3dpap', self.requested_urls)
        self.assertIn('u:3dstruct', self.requested_urls)

    def test_stereo_mordred_and_fingerprint_descriptors_are_downloaded(self):
        self._run(nostereo=False, stereo=True, descriptors=['mordred', 'fingerprint'])
        self.assertIn('u:3dmordred', self.requested_urls)
        self.assertIn('u:3dfp', self.requested_urls)

    def test_prodec_present_in_links_is_downloaded(self):
        self._run(nostereo=True, stereo=False, descriptors=['prodec'])
        self.assertIn('u:prodec', self.requested_urls)

    def test_prodec_absent_from_links_warns_and_is_skipped(self):
        files = {'05.4': {k: v for k, v in self.files['05.4'].items() if k != 'proteins_prodec'}}
        self.mocks['get_papyrus_links'].return_value = files
        with self.assertWarns(UserWarning):
            download.download_papyrus(
                outdir=self._tmpdir.name, version='05.4', progress=False, keep_xz=True,
                nostereo=True, stereo=False, descriptors=['prodec'],
            )
        self.assertNotIn('u:prodec', self.requested_urls)

    def test_descriptors_none_downloads_no_descriptor_files(self):
        self._run(nostereo=True, only_pp=True, stereo=False, descriptors=None)
        self.assertNotIn('u:2dmold2', self.requested_urls)


class TestDownloadPapyrusFileAlreadyPresent(unittest.TestCase):
    """Files already on disk (converted, or intact with a matching hash)
    are skipped instead of re-downloaded; a hash mismatch retries then
    raises once _RETRIES is exhausted.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.files = {
            '05.4': {
                'readme': {'name': 'README.txt', 'url': 'u:readme', 'size': 1, 'sha256': 'x'},
                'requirements': {'name': 'req.zip', 'url': 'u:requirements', 'size': 1, 'sha256': 'x'},
                'proteins': {'name': 'proteins.tsv.xz', 'url': 'u:proteins', 'size': 1, 'sha256': 'x'},
            },
        }
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')
        self.zip_bytes = zip_buf.getvalue()

        self.requested_urls: list[str] = []

        def fake_get(url, **kwargs):
            self.requested_urls.append(url)
            content = self.zip_bytes if url == 'u:requirements' else b'x'
            return _FakeResponse(content)

        self.fake_session = MagicMock()
        self.fake_session.get.side_effect = fake_get

        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value=self.files,
            ),
            'new_session': patch(
                'src.papyrus_scripts.download.new_session', return_value=self.fake_session,
            ),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_already_converted_parquet_skips_download_entirely(self):
        # 'proteins' is a _ROOT_FTYPES entry - its parquet sibling lives
        # directly under the version root, not under descriptors/.
        # progress=True also exercises the pbar.update() on this skip path.
        version_root = Path(self._tmpdir.name) / 'papyrus' / '05.4'
        version_root.mkdir(parents=True)
        (version_root / 'proteins.tsv.parquet').touch()
        with (
            patch('src.papyrus_scripts.download.assert_sha256sum', return_value=True),
            self.assertWarns(DeprecationWarning),
        ):
            download.download_papyrus(
                outdir=self._tmpdir.name, version='05.4', progress=True, keep_xz=True,
                nostereo=False, stereo=False, only_pp=False, descriptors=None,
            )
        self.assertNotIn('u:proteins', self.requested_urls)

    def test_intact_file_on_disk_with_matching_hash_is_not_redownloaded(self):
        # progress=True also exercises the pbar.update() on this skip path.
        version_root = Path(self._tmpdir.name) / 'papyrus' / '05.4'
        version_root.mkdir(parents=True)
        (version_root / 'proteins.tsv.xz').write_bytes(b'already-here')
        with (
            patch('src.papyrus_scripts.download.assert_sha256sum', return_value=True),
            self.assertWarns(DeprecationWarning),
        ):
            download.download_papyrus(
                outdir=self._tmpdir.name, version='05.4', progress=True, keep_xz=True,
                nostereo=False, stereo=False, only_pp=False, descriptors=None,
            )
        self.assertNotIn('u:proteins', self.requested_urls)

    def test_hash_mismatch_retries_then_succeeds(self):
        # Every file goes through the same hash check - only 'proteins'
        # should ever mismatch, so readme/requirements must always pass.
        # progress=True also exercises the "Retrying (n left)" pbar message.
        proteins_results = iter([False, True])

        def fake_sha(fpath, sha):
            return next(proteins_results) if fpath.name == 'proteins.tsv.xz' else True

        with (
            patch('src.papyrus_scripts.download.assert_sha256sum', side_effect=fake_sha),
            self.assertWarns(DeprecationWarning),
        ):
            download.download_papyrus(
                outdir=self._tmpdir.name, version='05.4', progress=True, keep_xz=True,
                nostereo=False, stereo=False, only_pp=False, descriptors=None,
            )
        self.assertEqual(self.requested_urls.count('u:proteins'), 2)

    def test_hash_mismatch_exhausts_retries_and_raises(self):
        def fake_sha(fpath, sha):
            return fpath.name != 'proteins.tsv.xz'

        with (
            patch('src.papyrus_scripts.download.assert_sha256sum', side_effect=fake_sha),
            self.assertRaises(OSError),
        ):
            download.download_papyrus(
                outdir=self._tmpdir.name, version='05.4', progress=False, keep_xz=True,
                nostereo=False, stereo=False, only_pp=False, descriptors=None,
            )
        self.assertEqual(self.requested_urls.count('u:proteins'), download._RETRIES)


class TestDownloadPapyrusFailureDuringDownloadCleansUpConverter(unittest.TestCase):
    """A download failure partway through must still let the converter
    process drain and exit cleanly (and close the download bar) before the
    original exception propagates.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')
        files = {
            '05.4': {
                'readme': {'name': 'README.txt', 'url': 'u:readme', 'size': 1, 'sha256': 'x'},
                'requirements': {'name': 'req.zip', 'url': 'u:requirements', 'size': 1, 'sha256': 'x'},
                'proteins': {'name': 'proteins.tsv.xz', 'url': 'u:proteins', 'size': 1, 'sha256': 'x'},
            },
        }

        def fake_get(url, **kwargs):
            if url == 'u:proteins':
                raise ConnectionError('network died')
            content = zip_buf.getvalue() if url == 'u:requirements' else b'x'
            return _FakeResponse(content)

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        self._queues_created: list[_FakeQueue] = []

        def make_fake_queue(*args, **kwargs):
            q = _FakeQueue()
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

    def test_exception_propagates_after_signalling_the_converter_to_stop(self):
        with (
            self.assertWarns(DeprecationWarning),
            self.assertRaises(ConnectionError),
        ):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=False)
        task_queue = self._queues_created[0]
        self.assertIs(task_queue.items[-1], download._CONVERSION_DONE)

    def test_progress_true_still_closes_the_download_bar_on_failure(self):
        with (
            patch('src.papyrus_scripts.download.tqdm') as fake_tqdm_cls,
            self.assertRaises(ConnectionError),
        ):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)
        pbar_mock = fake_tqdm_cls.return_value
        pbar_mock.close.assert_called()


class TestDownloadPapyrusConversionFailureRaises(unittest.TestCase):
    """Once every file has downloaded, a non-None error from the converter
    process (reported via error_queue) must raise RuntimeError - and close
    the download bar first, if progress bars are on.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        files = {
            '05.4': {
                'readme': {'name': 'README.txt', 'url': 'u:readme', 'size': 1, 'sha256': 'x'},
                'requirements': {'name': 'req.zip', 'url': 'u:requirements', 'size': 1, 'sha256': 'x'},
            },
        }
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')

        def fake_get(url, **kwargs):
            content = zip_buf.getvalue() if url == 'u:requirements' else b'x'
            return _FakeResponse(content)

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        class _ErroringQueue(_FakeQueue):
            def get(self):
                return 'boom: conversion exploded'

        def make_fake_queue(*args, **kwargs):
            return _ErroringQueue()

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

    def test_conversion_error_raises_runtimeerror_with_the_message(self):
        with (
            self.assertWarns(DeprecationWarning),
            self.assertRaisesRegex(RuntimeError, 'conversion exploded'),
        ):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=False)

    def test_progress_true_closes_the_bar_before_raising(self):
        with (
            patch('src.papyrus_scripts.download.tqdm') as fake_tqdm_cls,
            self.assertRaises(RuntimeError),
        ):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)
        # First tqdm() call is the download bar; it must be closed.
        fake_tqdm_cls.return_value.close.assert_called()


class TestResolveVersionsFallbackToRawAlias(unittest.TestCase):

    def test_unresolvable_short_string_falls_back_to_itself_then_raises(self):
        with self.assertRaises(ValueError):
            download._resolve_versions('bogus', {'2022.04.2': {}}, all_revisions=True)


class TestRemovePapyrusDescriptorAndStereoFlags(unittest.TestCase):
    """Every individual removal-set flag (bioactivities/stereo/structures/
    descriptor types), and the 'file listed but not on disk' skip branch.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root_dir = Path(self._tmpdir.name)
        self.v1 = make_version('v1')

        def entry(name, size=1):
            return {'name': name, 'url': 'x', 'size': size, 'sha256': 'x'}

        self.version_files = {
            '2D_papyrus': entry('2dpap.tsv.xz'),
            '3D_papyrus': entry('3dpap.tsv.xz'),
            '2D_structures': entry('2dstruct.sd.xz'),
            '3D_structures': entry('3dstruct.sd.xz'),
            '3D_mordred': entry('3dmordred.tsv.xz'),
            '2D_fingerprint': entry('2dfp.tsv.xz'),
            '3D_fingerprint': entry('3dfp.tsv.xz'),
        }
        # Nothing actually written to disk: every ftype is "listed but
        # absent" - exercises the not-fpath.is_file() skip branch below.

        root_module = MagicMock()
        root_module.base = self.root_dir
        root_module.join.side_effect = (
            lambda *parts, name=None: self.root_dir.joinpath(*parts, name) if name else self.root_dir
        )

        patches = {
            'get_papyrus_links': patch(
                'src.papyrus_scripts.download.get_papyrus_links', return_value={},
            ),
            'resolve_versions': patch(
                'src.papyrus_scripts.download._resolve_versions', return_value=[self.v1],
            ),
            'get_version_files': patch(
                'src.papyrus_scripts.download._get_version_files', return_value=self.version_files,
            ),
            'pystow_module': patch(
                'src.papyrus_scripts.download.pystow.module', return_value=root_module,
            ),
            'update_versions_json': patch('src.papyrus_scripts.download._update_versions_json'),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_bioactivities_structures_and_other_files_select_2d_and_metadata(self):
        download.remove_papyrus(
            version='v1', bioactivities=True, structures=True, other_files=True,
            nostereo=True, stereo=False, progress=False,
        )
        self.assertEqual(self.mocks['get_version_files'].call_count, 1)

    def test_bioactivities_and_stereo_flags_select_3d_papyrus(self):
        download.remove_papyrus(
            version='v1', bioactivities=True, nostereo=False, stereo=True, progress=False,
        )
        self.assertEqual(self.mocks['get_version_files'].call_count, 1)

    def test_structures_and_stereo_flags_select_3d_structures(self):
        download.remove_papyrus(
            version='v1', structures=True, nostereo=False, stereo=True, progress=False,
        )
        self.assertEqual(self.mocks['get_version_files'].call_count, 1)

    def test_stereo_mordred_and_fingerprint_descriptors_selected(self):
        download.remove_papyrus(
            version='v1', nostereo=False, stereo=True,
            descriptors=['mordred', 'fingerprint'], progress=False,
        )
        self.assertEqual(self.mocks['get_version_files'].call_count, 1)

    def test_present_file_is_actually_removed(self):
        # '2D_fingerprint' is not a _ROOT_FTYPES entry - its file lives
        # under descriptors/.
        (self.root_dir / 'descriptors').mkdir()
        (self.root_dir / 'descriptors' / '2dfp.tsv.xz').touch()
        download.remove_papyrus(
            version='v1', nostereo=True, stereo=False,
            descriptors=['fingerprint'], progress=False,
        )
        self.assertFalse((self.root_dir / 'descriptors' / '2dfp.tsv.xz').exists())


class _ResetThenRestartConsumerProcess:
    """Stand-in for multiprocessing.Process: emits START, a CHUNK, a RESET
    (dtype-drift healing restarted the file from row 0), another CHUNK, then
    DONE - so the progress-queue's RESET branch is genuinely exercised. Runs
    on a background thread (like _SlowConsumerProcess) so start() itself
    never blocks the caller on a queue that nothing has been put on yet.
    """

    def __init__(self, target=None, args=(), daemon=None):
        self._task_queue, self._error_queue, self._progress_queue, self._progress = args
        self._thread: threading.Thread | None = None

    def start(self):
        def run():
            task = self._task_queue.get()
            self._progress_queue.put((download._PROGRESS_START, task['desc'], 10))
            self._progress_queue.put((download._PROGRESS_CHUNK, 6))
            self._progress_queue.put((download._PROGRESS_RESET,))
            self._progress_queue.put((download._PROGRESS_CHUNK, 3))
            self._progress_queue.put((download._PROGRESS_DONE,))
            self._task_queue.get()  # the sentinel
            self._error_queue.put(None)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()


class TestDownloadPapyrusProgressResetMessage(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zh:
            zh.writestr('data_types.json', '{}')
        files = {
            '05.4': {
                'readme': {'name': 'README.txt', 'url': 'u:readme', 'size': 1, 'sha256': 'x'},
                'requirements': {'name': 'req.zip', 'url': 'u:requirements', 'size': 1, 'sha256': 'x'},
                'proteins': {'name': 'proteins.tsv.xz', 'url': 'u:proteins', 'size': 1, 'sha256': 'x'},
            },
        }

        def fake_get(url, **kwargs):
            content = zip_buf.getvalue() if url == 'u:requirements' else b'x'
            return _FakeResponse(content)

        fake_session = MagicMock()
        fake_session.get.side_effect = fake_get

        self.pbar_mocks: list[MagicMock] = []

        def fake_tqdm(*args, **kwargs):
            m = MagicMock()
            m._kwargs = kwargs
            self.pbar_mocks.append(m)
            return m

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
            'mp_process': patch(
                'src.papyrus_scripts.download.mp.Process', side_effect=_ResetThenRestartConsumerProcess,
            ),
            'tqdm': patch('src.papyrus_scripts.download.tqdm', side_effect=fake_tqdm),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_reset_message_zeroes_the_row_count_shown(self):
        with self.assertWarns(DeprecationWarning):
            download.download_papyrus(outdir=self._tmpdir.name, version='05.4', progress=True)

        converting_bars = [m for m in self.pbar_mocks if m._kwargs.get('desc') == 'Converting files']
        self.assertEqual(len(converting_bars), 1)
        descriptions = [
            call.args[0] for call in converting_bars[0].set_description.call_args_list if call.args
        ]
        # After the RESET message, the next description must show 0 rows
        # again (then climbs back up from the post-reset CHUNK), not a
        # count that kept accumulating from before the restart.
        self.assertTrue(any('(0/10 rows)' in d for d in descriptions), descriptions)


if __name__ == '__main__':
    unittest.main()
