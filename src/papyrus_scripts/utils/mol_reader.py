# -*- coding: utf-8 -*-

"""Molecular file readers with transparent decompression support."""

import bz2
import gzip
import io
import lzma
import re
import warnings
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    ForwardSDMolSupplier,
    MaeMolSupplier,
    MolFromMol2Block,
    SmilesMolSupplier,
    SmilesMolSupplierFromText,
)
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A source accepted by any supplier: a file path or an open file-like object.
Source = Union[str, Path, io.TextIOBase, io.BufferedIOBase]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#: Maps file extension → (compression label, open-function, binary?)
_COMPRESSION_MAP: dict = {
    '.xz':  ('lzma', lzma.open, True),
    '.gz':  ('zlib', gzip.open, True),
    '.bz2': ('bz2',  bz2.open,  True),
}

#: Maps molecular format extension → format label.
_FORMAT_MAP: dict = {
    '.smi':  'smi',
    '.mae':  'mae',
    '.sd':   'sd',
    '.sdf':  'sd',
    '.mol2': 'mol2',
    '.mol':  'mol',
}

#: Format labels that need a binary stream from the underlying open function.
_BINARY_FORMATS = {'mae', 'sd', 'mol'}

#: Format labels that need a text stream.
_TEXT_FORMATS = {'smi', 'mol2'}


def _strip_compression_suffix(filename: str) -> Tuple[Optional[str], Callable, bool, str]:
    """Return ``(label, open_fn, is_binary, inner_filename)`` for a (possibly
    compressed) filename.

    :param filename: the full filename, e.g. ``'data.sd.xz'``
    :returns: compression label (or ``None``), the matching ``open`` function,
        whether the decompressed stream is binary, and the filename with the
        compression extension removed.
    """
    for ext, (label, open_fn, binary) in _COMPRESSION_MAP.items():
        if filename.endswith(ext):
            # Use str.removesuffix to avoid the rstrip character-set pitfall.
            return label, open_fn, binary, filename.removesuffix(ext)
    return None, open, False, filename


def _detect_format(filename: str) -> str:
    """Infer the molecular format from *filename* (extension-based).

    :param filename: filename after compression suffix has been stripped
    :raises ValueError: if the extension is not recognised
    """
    for ext, label in _FORMAT_MAP.items():
        if filename.endswith(ext):
            return label
    raise ValueError(
        f'Cannot infer molecular format from filename {filename!r}. '
        f'Supported extensions: {list(_FORMAT_MAP)}'
    )


# ---------------------------------------------------------------------------
# ForwardMol2MolSupplier
# ---------------------------------------------------------------------------

class ForwardMol2MolSupplier:
    """A forward (streaming) Mol2 molecule supplier.

    Accepts either a filename or an already-open **text-mode** file object.
    Binary streams (e.g. from ``lzma.open(..., 'rb')``) are not supported
    because ``MolFromMol2Block`` requires a string, not bytes.

    Molecules are yielded one at a time; the file is split on the
    ``@<TRIPOS>MOLECULE`` record delimiter without loading it entirely into
    memory.
    """

    _DELIMITER = '@<TRIPOS>MOLECULE'
    _BUFFER_SIZE = 32_768  # 32 kB

    def __init__(
        self,
        fileobj: Union[str, Path, io.TextIOBase],
        sanitize: bool = True,
        removeHs: bool = True,
        cleanupSubstructures: bool = True,
    ):
        """Initialise the supplier.

        :param fileobj: path to a ``.mol2`` file, or an open text-mode stream
        :param sanitize: passed to ``MolFromMol2Block``
        :param removeHs: passed to ``MolFromMol2Block``
        :param cleanupSubstructures: passed to ``MolFromMol2Block``
        """
        self.sanitize = sanitize
        self.removeHs = removeHs
        self.cleanupSubstructures = cleanupSubstructures
        self._buffer: str = ''

        if isinstance(fileobj, (str, Path)):
            self._owns_handle = True
            self._handle: io.TextIOBase = open(fileobj, 'r')
        elif isinstance(fileobj, io.TextIOBase):
            self._owns_handle = False
            self._handle = fileobj
        else:
            raise TypeError(
                'ForwardMol2MolSupplier requires a filename or a text-mode '
                f'file object, got {type(fileobj).__name__!r}.'
            )

        self._iterator: Optional[Iterator] = None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _iterate(self) -> Iterator[Optional[Chem.Mol]]:
        self._buffer = self._handle.read(self._BUFFER_SIZE)
        while True:
            # Find all delimiter positions that are NOT at position 0
            # (position 0 means the current buffer starts with a new record,
            # which we have not finished reading yet).
            sep_positions = [
                m.start()
                for m in re.finditer(re.escape(self._DELIMITER), self._buffer)
                if m.start() != 0
            ]

            if not sep_positions:
                # Need more data to find the next record boundary.
                more = self._handle.read(self._BUFFER_SIZE)
                if more:
                    self._buffer += more
                else:
                    # EOF — emit whatever remains as the last record.
                    if self._buffer.strip():
                        yield self._parse_block(self._buffer)
                    break
            else:
                # Emit everything up to the first separator as one record.
                yield self._parse_block(self._buffer[: sep_positions[0]])
                self._buffer = self._buffer[sep_positions[0]:]

    def _parse_block(self, block: str) -> Optional[Chem.Mol]:
        return MolFromMol2Block(
            block,
            self.sanitize,
            self.removeHs,
            self.cleanupSubstructures,
        )

    def __iter__(self) -> Iterator[Optional[Chem.Mol]]:
        if self._iterator is None:
            self._iterator = self._iterate()
        yield from self._iterator

    def __next__(self) -> Optional[Chem.Mol]:
        if self._iterator is None:
            self._iterator = self._iterate()
        return next(self._iterator)

    def close(self) -> None:
        """Close the underlying file handle if it was opened by this supplier."""
        if self._owns_handle:
            self._handle.close()


# ---------------------------------------------------------------------------
# ForwardSmilesMolSupplier
# ---------------------------------------------------------------------------

class ForwardSmilesMolSupplier:
    """A forward (streaming) SMILES molecule supplier.

    Accepts either a filename (handled by RDKit's ``SmilesMolSupplier``
    directly) or an already-open **text-mode** file object (streamed manually
    line-by-line via ``SmilesMolSupplierFromText``).
    """

    _BUFFER_SIZE = 32_768  # 32 kB

    def __init__(
        self,
        fileobj: Union[str, Path, io.TextIOBase],
        delimiter: str = '\t',
        smilesColumn: int = 0,
        nameColumn: int = 1,
        titleLine: bool = True,
        sanitize: bool = True,
    ):
        """Initialise the supplier.

        :param fileobj: path to a ``.smi`` file, or an open text-mode stream
        :param delimiter: column delimiter
        :param smilesColumn: zero-based index of the SMILES column
        :param nameColumn: zero-based index of the name column
        :param titleLine: whether the first line is a header
        :param sanitize: whether to sanitise molecules
        """
        self.delimiter    = delimiter
        self.smilesColumn = smilesColumn
        self.nameColumn   = nameColumn
        self.titleLine    = titleLine
        self.sanitize     = sanitize
        self._buffer: str = ''
        self._iterator: Optional[Iterator] = None

        if isinstance(fileobj, (str, Path)):
            # Let RDKit handle the file natively — most efficient path.
            self._handle = None
            self._owns_handle = False
            self._iterator = iter(
                SmilesMolSupplier(
                    str(fileobj),
                    self.delimiter,
                    self.smilesColumn,
                    self.nameColumn,
                    self.titleLine,
                    self.sanitize,
                )
            )
        elif isinstance(fileobj, io.TextIOBase):
            self._handle = fileobj
            self._owns_handle = False   # caller owns it
        else:
            raise TypeError(
                'ForwardSmilesMolSupplier requires a filename or a text-mode '
                f'file object, got {type(fileobj).__name__!r}.'
            )

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _iterate(self) -> Iterator[Optional[Chem.Mol]]:
        """Stream molecules from a text-mode file object line by line."""
        if self.titleLine:
            self._handle.readline()

        self._buffer = self._handle.read(self._BUFFER_SIZE)
        while True:
            newline_positions = [m.start() for m in re.finditer('\n', self._buffer)]

            if not newline_positions:
                more = self._handle.read(self._BUFFER_SIZE)
                if more:
                    self._buffer += more
                else:
                    # EOF — emit whatever is left.
                    if self._buffer.strip():
                        yield self._parse_smiles_line(self._buffer)
                    break
            else:
                end = newline_positions[0] + 1  # include the newline
                yield self._parse_smiles_line(self._buffer[:end])
                self._buffer = self._buffer[end:]

    def _parse_smiles_line(self, line: str) -> Optional[Chem.Mol]:
        # Suppress RDKit log noise when there is no name column.
        RDLogger.DisableLog('rdApp.*')
        mol = next(
            SmilesMolSupplierFromText(
                line,
                self.delimiter,
                self.smilesColumn,
                self.nameColumn,
                False,       # titleLine=False — we already handled it
                self.sanitize,
            )
        )
        RDLogger.EnableLog('rdApp.*')
        return mol

    def __iter__(self) -> Iterator[Optional[Chem.Mol]]:
        if self._iterator is None:
            self._iterator = self._iterate()
        yield from self._iterator

    def __next__(self) -> Optional[Chem.Mol]:
        if self._iterator is None:
            self._iterator = self._iterate()
        return next(self._iterator)

    def close(self) -> None:
        """No-op; file handle is always owned by the caller or MolSupplier."""
        pass


# ---------------------------------------------------------------------------
# MolSupplier
# ---------------------------------------------------------------------------

class MolSupplier:
    """Unified molecule supplier with automatic format and compression detection.

    Wraps RDKit's individual suppliers behind a single interface that:

    * detects compression from the filename suffix (``.xz``, ``.gz``, ``.bz2``)
    * detects the molecular format from the inner filename suffix
    * opens the decompressed stream in the correct mode (text vs. binary) for
      the chosen format
    * iterates as ``(mol_id, Chem.Mol)`` pairs, optionally with a progress bar
    * is usable as a context manager

    Usage::

        with MolSupplier('data.sd.xz', show_progress=True, total=50_000) as sup:
            for mol_id, rdmol in sup:
                ...
    """

    VALID_FORMATS     = frozenset(_FORMAT_MAP.values())
    VALID_COMPRESSION = frozenset(label for label, *_ in _COMPRESSION_MAP.values())

    def __init__(
        self,
        source: Optional[Source] = None,
        supplier: Optional[Iterable[Chem.Mol]] = None,
        format: Optional[str] = None,
        compression: Optional[str] = None,
        **kwargs,
    ):
        """Initialise the supplier.

        Exactly one of *source* or *supplier* must be provided.

        :param source: a filename / :class:`~pathlib.Path`, or an open
            file-like object.  When a filename is given, format and compression
            are detected automatically.  When a file-like object is given,
            *format* must be provided explicitly.
        :param supplier: a pre-built RDKit-compatible molecule supplier.
            When given, *source*, *format*, and *compression* are ignored.
        :param format: molecular format; one of
            ``'smi'``, ``'mae'``, ``'sd'``, ``'mol2'``, ``'mol'``.
            Required when *source* is a file-like object.
        :param compression: compression type; one of ``'lzma'``, ``'zlib'``,
            ``'bz2'``.  Detected automatically when *source* is a filename.
        :param kwargs: additional keyword arguments forwarded to the underlying
            RDKit supplier.  Three special keys are consumed here:

            * ``start_id`` *(int, default 0)* – starting molecule index
            * ``total`` *(int, default None)* – total molecules for the
              progress bar
            * ``show_progress`` *(bool, default False)* – display a tqdm bar
        :raises ValueError: if neither or both of *source* / *supplier* are
            given, or if a file-like object is provided without *format*.
        """
        if source is None and supplier is None:
            raise ValueError('Exactly one of `source` or `supplier` must be provided.')

        # Extract iteration-control kwargs before forwarding the rest.
        self._iter_start    = kwargs.pop('start_id',      0)
        self._iter_total    = kwargs.pop('total',         None)
        self._iter_progress = kwargs.pop('show_progress', False)
        self._supplier_kwargs = kwargs

        self._owns_handle = False   # whether we opened the stream
        self._handle: Optional[io.IOBase] = None
        self._inner_supplier: Optional[Iterable[Chem.Mol]] = None
        self._iterator: Optional[Iterator[Tuple[int, Chem.Mol]]] = None

        # ------------------------------------------------------------------
        # Branch 1: a pre-built supplier is given — nothing else to do.
        # ------------------------------------------------------------------
        if supplier is not None:
            self._inner_supplier = supplier
            self.format      = None
            self.compression = None
            return

        # ------------------------------------------------------------------
        # Branch 2: source is a filename or Path.
        # ------------------------------------------------------------------
        if isinstance(source, (str, Path)):
            filename = str(source)
            self._owns_handle = True

            # --- Detect / validate compression ---
            if compression is not None:
                if compression not in self.VALID_COMPRESSION:
                    raise ValueError(
                        f'compression must be one of {sorted(self.VALID_COMPRESSION)}, '
                        f'got {compression!r}'
                    )
                self.compression = compression
                open_fn, is_binary = self._open_fn_for_label(compression)
                # Still derive the truncated filename (compression suffix stripped)
                # so format auto-detection below works even when compression is
                # passed explicitly instead of being inferred from the filename.
                _, _, _, inner_filename = _strip_compression_suffix(filename)
            else:
                label, open_fn, is_binary, inner_filename = _strip_compression_suffix(filename)
                self.compression = label

            # --- Detect / validate format ---
            if format is not None:
                if format not in self.VALID_FORMATS:
                    raise ValueError(
                        f'format must be one of {sorted(self.VALID_FORMATS)}, '
                        f'got {format!r}'
                    )
                self.format = format
            else:
                self.format = _detect_format(inner_filename)

            # --- Open stream in the correct mode ---
            stream_mode = self._stream_mode(self.format, is_binary)
            self._handle = open_fn(filename, stream_mode)
            self._inner_supplier = self._make_supplier(self.format, self._handle)
            return

        # ------------------------------------------------------------------
        # Branch 3: source is an already-open file-like object.
        # ------------------------------------------------------------------
        if isinstance(source, (io.TextIOBase, io.BufferedIOBase)):
            if format is None:
                raise ValueError(
                    '`format` must be specified when `source` is a file-like object.'
                )
            if format not in self.VALID_FORMATS:
                raise ValueError(
                    f'format must be one of {sorted(self.VALID_FORMATS)}, '
                    f'got {format!r}'
                )
            self.format      = format
            self.compression = None     # unknown / irrelevant
            self._handle     = source   # we do NOT own this handle
            self._inner_supplier = self._make_supplier(format, source)
            return

        raise TypeError(
            '`source` must be a filename, a Path, or an open file-like object, '
            f'got {type(source).__name__!r}.'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_fn_for_label(label: str) -> Tuple[Callable, bool]:
        """Return ``(open_fn, is_binary)`` for an explicit compression label."""
        for _label, open_fn, binary in _COMPRESSION_MAP.values():
            if _label == label:
                return open_fn, binary
        raise ValueError(f'Unknown compression label {label!r}')

    @staticmethod
    def _stream_mode(fmt: str, stream_is_binary: bool) -> str:
        """Return the open-mode string appropriate for *fmt* and the stream kind.

        SD / MAE / MOL require a binary stream from rdkit.
        SMILES / Mol2 suppliers need a text stream.

        :param fmt: molecular format label
        :param stream_is_binary: whether the decompressor returns bytes
        """
        if fmt in _BINARY_FORMATS:
            return 'rb'
        # Text formats — ensure we get a str stream.
        # lzma/gzip/bz2 default to binary; pass 'rt' to get text.
        return 'rt' if stream_is_binary else 'r'

    def _make_supplier(
        self,
        fmt: str,
        handle: io.IOBase,
    ) -> Iterable[Chem.Mol]:
        """Instantiate the correct low-level RDKit supplier for *fmt*.

        :param fmt: molecular format label
        :param handle: open file handle (text or binary as required by *fmt*)
        """
        kw = self._supplier_kwargs
        if fmt == 'smi':
            return ForwardSmilesMolSupplier(handle, **kw)
        if fmt == 'mae':
            return MaeMolSupplier(handle, **kw)
        if fmt in ('sd', 'mol'):
            return ForwardSDMolSupplier(handle, **kw)
        if fmt == 'mol2':
            # ForwardMol2MolSupplier needs a text-mode handle.
            if isinstance(handle, io.BufferedIOBase):
                raise TypeError(
                    'Mol2 format requires a text-mode stream. '
                    'Use an uncompressed file or open with mode="rt".'
                )
            return ForwardMol2MolSupplier(handle, **kw)
        raise ValueError(f'Unsupported format {fmt!r}')

    # ------------------------------------------------------------------
    # Iteration-control setter (kept for backwards compatibility)
    # ------------------------------------------------------------------

    def set_start_progress_total(
        self,
        start: int = 0,
        progress: bool = True,
        total: Optional[int] = None,
    ) -> None:
        """Configure iteration behaviour after construction.

        :param start: starting molecule index (used in ``(mol_id, mol)`` pairs)
        :param progress: show a tqdm progress bar
        :param total: total number of molecules (for the progress bar)
        """
        self._iter_start    = start
        self._iter_progress = progress
        self._iter_total    = total

    # ------------------------------------------------------------------
    # Core generator
    # ------------------------------------------------------------------

    def _processed_mol_supplier(self) -> Iterator[Tuple[int, Chem.Mol]]:
        """Yield ``(mol_id, rdmol)`` pairs, skipping ``None`` entries."""
        enumerated = enumerate(self._inner_supplier, self._iter_start)
        if self._iter_progress:
            enumerated = tqdm(enumerated, total=self._iter_total, ncols=100)

        for mol_id, rdmol in enumerated:
            if rdmol is not None:
                yield mol_id, rdmol
            else:
                warnings.warn(
                    f'Molecule at index {mol_id} could not be parsed and was skipped.',
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Iteration protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[int, Chem.Mol]]:
        if self._iterator is None:
            self._iterator = self._processed_mol_supplier()
        yield from self._iterator

    def __next__(self) -> Tuple[int, Chem.Mol]:
        if self._iterator is None:
            self._iterator = self._processed_mol_supplier()
        return next(self._iterator)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying file handle (if opened by this supplier)."""
        # Let the inner supplier clean up first (ForwardSmiles/Mol2 have close()).
        if hasattr(self._inner_supplier, 'close'):
            self._inner_supplier.close()
        self._inner_supplier = None

        if self._owns_handle and self._handle is not None:
            self._handle.close()
        self._handle = None
