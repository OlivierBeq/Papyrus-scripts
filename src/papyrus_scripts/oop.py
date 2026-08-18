# -*- coding: utf-8 -*-

"""Object-oriented API for the Papyrus dataset."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import prodec
import pystow

from . import download, preprocess, reader, subsim_search
from .fingerprint import Fingerprint, MorganFingerprint
from .matchRCSB import get_matches as get_pdb_matches
from .utils import IO

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_papyrus_version(version: str | IO.PapyrusVersion) -> IO.PapyrusVersion:
    """Return a :class:`~utils.IO.PapyrusVersion` from *version*.

    If *version* is already a :class:`~utils.IO.PapyrusVersion` it is returned
    unchanged, avoiding the ``AttributeError`` that would be raised by calling
    ``PapyrusVersion(version=<PapyrusVersion>)``.

    :param version: raw version string or an existing :class:`~utils.IO.PapyrusVersion`
    """
    if isinstance(version, IO.PapyrusVersion):
        return version
    return IO.PapyrusVersion(version=version)


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return ⌈numerator / denominator⌉ without floating-point arithmetic."""
    return -(-numerator // denominator)


def _num_chunks(num_rows: int | None, chunksize: int | None) -> int | None:
    """Return the number of chunks or ``None`` when *chunksize* is ``None``."""
    if chunksize is None or num_rows is None:
        return None
    return _ceil_div(num_rows, chunksize)


def _id_column(is3d: bool) -> str:
    """Return the molecule-identifier column name for the given stereo flag."""
    return 'InChIKey' if is3d else 'connectivity'


#: Exceptions signalling "this isn't available locally yet" - the trigger to
#: download it, throughout this module's various "try to read; download and
#: retry on a cache miss" call sites. FileNotFoundError/NotADirectoryError
#: come from utils.IO.locate_file when a file is genuinely missing; KeyError
#: from get_num_rows_in_file's data_size.json lookup. OSError/ValueError
#: come from utils.IO.process_data_version (reached via every reader.py
#: function through _resolve_version) - its own two documented exceptions
#: for "no Papyrus data downloaded at all" and "this specific version isn't
#: among the downloaded ones" respectively. Missing these two here used to
#: crash instead of downloading the very first time a given version was
#: ever requested on a machine that already had a *different* version
#: downloaded (get_downloaded_versions returns non-empty, so the OSError
#: branch is skipped, but the requested version itself isn't in that list).
_NOT_AVAILABLE_LOCALLY: tuple[type[Exception], ...] = (
    FileNotFoundError, NotADirectoryError, KeyError, OSError, ValueError,
)


# ---------------------------------------------------------------------------
# Deferred loading
# ---------------------------------------------------------------------------
#
# A freshly-constructed PapyrusDataset must not download (or even check for)
# its bioactivity/protein files until data is actually needed - not at
# construction, and not when a filter (keep_quality, keep_accession, …) is
# applied, only once aggregate()/agg()/consume_chunks()/to_dataframe() (or
# an equivalent - .proteins(), .match_rcsb_pdb(), FPSubSim2-backed filters)
# forces materialisation. Every filter method just wraps the *unresolved*
# state in one more pending operation instead.

class _Deferred:
    """A value computed by *loader* the first time it's needed, then cached."""

    __slots__ = ('_loader', '_value', '_resolved')

    def __init__(self, loader: Callable[[], Any]) -> None:
        self._loader = loader
        self._value: Any = None
        self._resolved = False

    def get(self) -> Any:
        if not self._resolved:
            self._value = self._loader()
            self._resolved = True
        return self._value


def _resolve(value: Any) -> Any:
    """Return *value* itself, or the result of calling it if it's a :class:`_Deferred`."""
    return value.get() if isinstance(value, _Deferred) else value


class _LazyBioactivity:
    """An unresolved bioactivity stream, plus pending filter methods to apply once loaded.

    Immutable - :meth:`then` returns a new instance, so branching a filter
    chain (applying two different filters to the same starting point) never
    lets one branch's cached result leak into the other's.
    """

    __slots__ = ('_loader', '_ops', '_value', '_resolved')

    def __init__(self, loader: Callable[[], Any], ops: tuple[Callable[[Any], Any], ...] = ()) -> None:
        self._loader = loader
        self._ops = ops
        self._value: Any = None
        self._resolved = False

    def then(self, op: Callable[[Any], Any]) -> _LazyBioactivity:
        """Return a new :class:`_LazyBioactivity` with *op* appended to the pending queue."""
        return _LazyBioactivity(self._loader, self._ops + (op,))

    def resolve(self) -> Any:
        """Load the base data (downloading it first if needed) and apply every pending op, once."""
        if not self._resolved:
            data = self._loader()
            for op in self._ops:
                data = op(data)
            self._value = data
            self._resolved = True
        return self._value


def _apply_lazy(data: Any, fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Return ``fn(data=data, **kwargs)``, deferred if *data* is still a :class:`_LazyBioactivity`.

    Any :class:`_Deferred` value in *kwargs* (e.g. ``protein_data`` for
    :func:`~preprocess.keep_protein_class`) is resolved at the same time as
    *data* - immediately if *data* is already resolved, otherwise only once
    the returned pending operation actually runs.
    """
    if isinstance(data, _LazyBioactivity):
        return data.then(lambda d: fn(data=d, **{k: _resolve(v) for k, v in kwargs.items()}))
    return fn(data=data, **{k: _resolve(v) for k, v in kwargs.items()})


class _PapyrusSource:
    """Downloads (if needed) and reads the bioactivity/protein files for one ``(version, is3d, plusplus)`` selection.

    Loaded at most once, however many :class:`PapyrusDataset`/
    :class:`_LazyBioactivity` instances in a filter chain need them.

    Also the single place that collects, across an entire filter chain, which
    descriptor sets and/or molecular structures will be needed downstream
    (via :meth:`request_descriptors`/:meth:`request_structures`, called by
    :meth:`PapyrusDataset.molecular_descriptors`/:meth:`PapyrusDataset.molecules`
    and their :class:`PapyrusMoleculeSet` counterparts while the chain is
    still being built, i.e. before any ``.agg()``). Since one
    :class:`_PapyrusSource` instance is shared by reference across every
    :class:`PapyrusDataset` derived from the same root (threaded through
    ``papyrus_params['_source']``), by the time materialisation actually
    starts every requirement registered anywhere in the chain is already
    known - so :meth:`_ensure_loaded` can issue one combined
    :func:`~download.download_papyrus` call covering everything instead of
    one separate download-and-convert cycle per file type as each is first
    touched.
    """

    def __init__(
            self,
            pv: IO.PapyrusVersion,
            is3d: bool,
            plusplus: bool,
            chunksize: int | None,
            source_path: str | Path | None,
            download_progress: bool,
            keep_original_files: bool,
            disk_margin: float,
    ) -> None:
        self._pv = pv
        self._is3d = is3d
        self._plusplus = plusplus
        self._chunksize = chunksize
        self._source_path = source_path
        self._download_progress = download_progress
        self._keep_original_files = keep_original_files
        self._disk_margin = disk_margin
        self._bioactivity: pl.DataFrame | pl.LazyFrame | None = None
        self._proteins: pl.DataFrame | None = None
        self._num_rows: int | None = None
        self._loaded = False
        self._descriptor_types: set[str] = set()
        self._need_structures: bool = False

    def request_descriptors(self, desc_type: str) -> None:
        """Register *desc_type* as needed by this chain's next combined download.

        Safe to call more than once (e.g. two branches of the same chain
        requesting different descriptor sets) - every distinct type
        requested anywhere in the chain ends up in the same combined
        download.

        :param desc_type: descriptor set, as accepted by
            :meth:`PapyrusDataset.molecular_descriptors`
        """
        self._descriptor_types.add(desc_type)

    def request_structures(self) -> None:
        """Register molecular structures as needed by this chain - see :meth:`request_descriptors`."""
        self._need_structures = True

    def _load(self) -> tuple[int, pl.DataFrame | pl.LazyFrame, pl.DataFrame]:
        num_rows = IO.get_num_rows_in_file(
            filetype='bioactivities', is3D=self._is3d,
            version=self._pv, plusplus=self._plusplus,
            root_folder=self._source_path,
        )
        bioactivity_data = reader.read_papyrus(
            is3d=self._is3d, version=self._pv, plusplus=self._plusplus,
            chunksize=self._chunksize, source_path=self._source_path,
        )
        protein_data = reader.read_protein_set(
            source_path=self._source_path, version=self._pv,
        )
        return num_rows, bioactivity_data, protein_data

    def _missing_registered_extras(self) -> bool:
        """Whether any descriptor set / structures registered so far is not yet on disk.

        Any failure to even determine that (e.g. this version isn't
        recognised as downloaded locally at all yet) is treated the same as
        "missing" - the safe default is to let the combined download below
        run and sort it out, rather than silently skipping something the
        chain will need.
        """
        try:
            if self._need_structures and not reader.molecular_structures_available(
                is3d=self._is3d, version=self._pv, source_path=self._source_path,
            ):
                return True
            return any(
                not reader.molecular_descriptors_available(
                    desc_type=desc_type, is3d=self._is3d,
                    version=self._pv, source_path=self._source_path,
                )
                for desc_type in self._descriptor_types
            )
        except _NOT_AVAILABLE_LOCALLY:
            return True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Download the bioactivity file matching (is3d, plusplus), the
        # always-needed protein-target file, and every descriptor
        # set/structures requested anywhere in the chain so far (see
        # request_descriptors/request_structures) - all in one combined
        # call, rather than downloading bioactivity+proteins here and
        # leaving descriptors/structures to trigger their own separate
        # download cycle later when first read. See PapyrusDataset.__init__
        # for why is_local_version_available can't gate this.
        try:
            self._num_rows, self._bioactivity, self._proteins = self._load()
            need_download = self._missing_registered_extras()
        except _NOT_AVAILABLE_LOCALLY:
            need_download = True
        if need_download:
            download.download_papyrus(
                outdir=self._source_path,
                # pystow_path_key (not .version): see PapyrusDataset.__init__.
                version=self._pv.pystow_path_key,
                nostereo=not self._is3d, stereo=self._is3d, only_pp=self._plusplus,
                structures=self._need_structures,
                descriptors=sorted(self._descriptor_types) or None,
                progress=self._download_progress, disk_margin=self._disk_margin,
                keep_xz=self._keep_original_files,
            )
            self._num_rows, self._bioactivity, self._proteins = self._load()
        self._loaded = True

    def get_bioactivity(self) -> pl.DataFrame | pl.LazyFrame:
        self._ensure_loaded()
        if self._bioactivity is None:
            raise RuntimeError('bioactivity not loaded despite _ensure_loaded()')
        return self._bioactivity

    def get_proteins(self) -> pl.DataFrame:
        self._ensure_loaded()
        if self._proteins is None:
            raise RuntimeError('proteins not loaded despite _ensure_loaded()')
        return self._proteins

    def get_num_rows(self) -> int:
        self._ensure_loaded()
        if self._num_rows is None:
            raise RuntimeError('num_rows not set despite _ensure_loaded()')
        return self._num_rows


# ---------------------------------------------------------------------------
# PapyrusDataset
# ---------------------------------------------------------------------------

class PapyrusDataset:
    """Papyrus dataset — the main entry point for data access and filtering.

    Every filter method (``keep_*``, ``contains``, ``isin``, …) returns a new
    :class:`PapyrusDataset` whose data stream is lazily filtered.  Call
    :meth:`aggregate` (or its aliases :meth:`agg`, :meth:`to_dataframe`,
    :meth:`consume_chunks`) to materialise the result into a DataFrame.
    """

    papyrus_bioactivity_data: pd.DataFrame | pl.DataFrame | pl.LazyFrame | Iterator[pd.DataFrame] | _LazyBioactivity
    papyrus_protein_data: pd.DataFrame | pl.DataFrame | _Deferred
    papyrus_params: dict

    def __init__(
            self,
            version: str | IO.PapyrusVersion = 'latest',
            is3d: bool = False,
            plusplus: bool = True,
            chunksize: int | None = 1_000_000,
            source_path: str | Path | None = None,
            download_progress: bool = True,
            keep_original_files: bool = False,
            disk_margin: float = 0.10,
    ) -> None:
        """Read, filter and aggregate data from a release of the Papyrus dataset.

        :param version: dataset version to use; either a
            :class:`~utils.IO.PapyrusVersion` or a string accepted by
            :class:`~utils.IO.PapyrusVersion` (default: ``'latest'``)
        :param is3d: load the stereochemistry-aware (3D) lower-quality data
            (default: False)
        :param plusplus: load the Papyrus++ curated high-quality subset
            (default: True)
        :param chunksize: rows per chunk when reading; ``None`` reads
            everything at once (not recommended for large datasets; default:
            1 000 000)
        :param source_path: root directory for Papyrus data (default:
            pystow's home directory)
        :param download_progress: show download progress bars when data is
            not yet on disk
        :param keep_original_files: keep the downloaded ``.tsv.xz`` originals
            after converting them to Parquet (default: False - every tabular
            file downloaded for this dataset, and later for any descriptor
            fetched lazily through it, is converted to Parquet and the
            ``.xz`` original is deleted, matching :func:`~download.download_papyrus`)
        :param disk_margin: safety margin passed to
            :func:`~download.download_papyrus` for any download this
            dataset triggers (default: 0.10)
        :raises ValueError: if both *is3d* and *plusplus* are True (the 3D
            Papyrus++ combination does not exist)
        """
        pv = _ensure_papyrus_version(version)
        source = _PapyrusSource(
            pv, is3d, plusplus, chunksize, source_path,
            download_progress, keep_original_files, disk_margin,
        )

        # Nothing is downloaded, converted, or even checked for here - only
        # once aggregate()/agg()/consume_chunks()/to_dataframe() (or an
        # equivalent - .proteins(), .match_rcsb_pdb(), a similarity-search
        # filter, keep_protein_class()/keep_organism()) actually needs the
        # data does _PapyrusSource resolve it, downloading only the
        # bioactivity file matching (is3d, plusplus) plus the always-needed
        # protein-target file - not every stereo/++ combination, structures,
        # or descriptors regardless of whether this dataset ever uses them.
        # is_local_version_available (whether *anything* was ever downloaded
        # for this version) can't gate this: a version downloaded before
        # under different (is3d, plusplus) choices would wrongly look
        # "available" while still missing the specific file needed now, so
        # _PapyrusSource reads first and only downloads on a genuine cache
        # miss for this exact selection.
        #
        # Descriptors/structures chained in before that first materialisation
        # point (.molecular_descriptors(...), .molecules(...) - see their
        # request_descriptors()/request_structures() calls into `source`) are
        # folded into that same combined download instead of triggering their
        # own separate download-and-convert cycle once they're first read -
        # `source` is shared by reference across the whole filter chain
        # (threaded through papyrus_params['_source']), so every requirement
        # registered anywhere in the chain is known by the time _PapyrusSource
        # actually resolves. Requesting one only *after* the chain has already
        # been materialised once (e.g. calling .agg() first, then
        # .molecular_descriptors(...).agg() on the same dataset afterwards)
        # still triggers its own separate download at that later point - there
        # is no way to know about a requirement that hasn't been expressed yet.
        self.papyrus_params: dict = dict(
            is3d=is3d,
            version=pv,
            plusplus=plusplus,
            chunksize=chunksize,
            source_path=source_path,
            download_progress=download_progress,
            keep_original_files=keep_original_files,
            disk_margin=disk_margin,
            _source=source,
        )
        self.papyrus_bioactivity_data = _LazyBioactivity(loader=source.get_bioactivity)
        self.papyrus_protein_data = _Deferred(loader=source.get_proteins)
        self._fpsubsim2_: FPSubSim2Engine | None = None
        self._can_reset: bool = True

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def from_dataframe(
            df: pd.DataFrame | pl.DataFrame,
            is3d: bool,
            version: str | IO.PapyrusVersion,
            plusplus: bool = True,
            source_path: str | Path | None = None,
            download_progress: bool = True,
            chunksize: int | None = None,
            keep_original_files: bool = False,
            disk_margin: float = 0.10,
    ) -> PapyrusDataset:
        """Create a :class:`PapyrusDataset` from an existing DataFrame.

        :param df: DataFrame of Papyrus bioactivity data (all standard columns
            must be present)
        :param is3d: whether *df* was derived from the 3D (stereo) dataset
        :param version: version of the dataset *df* was obtained from
        :param plusplus: whether *df* was derived from the Papyrus++ subset
        :param source_path: root directory for Papyrus data
        :param download_progress: whether download progress was shown
        :param chunksize: chunk size to record in ``papyrus_params``
        :param keep_original_files: keep ``.tsv.xz`` originals after
            conversion to Parquet for any descriptor later downloaded
            lazily through this dataset (default: False)
        :param disk_margin: safety margin for any download later triggered
            through this dataset (default: 0.10)
        :returns: a :class:`PapyrusDataset` wrapping *df*
        """
        pv = _ensure_papyrus_version(version)
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        dataset = PapyrusDataset.__new__(PapyrusDataset)
        dataset.papyrus_bioactivity_data = df
        dataset.papyrus_protein_data = reader.read_protein_set(
            source_path=source_path, version=pv,
        )
        dataset.papyrus_params = dict(
            is3d=is3d, version=pv, plusplus=plusplus,
            chunksize=chunksize, source_path=source_path,
            num_rows=len(df), download_progress=download_progress,
            keep_original_files=keep_original_files,
            disk_margin=disk_margin,
        )
        dataset._fpsubsim2_ = None
        dataset._can_reset = False
        return dataset

    @staticmethod
    def _from_data(
            papyrus_bioactivity_data: Iterator[pd.DataFrame] | pd.DataFrame,
            papyrus_protein_data: pd.DataFrame,
            papyrus_params: dict,
    ) -> PapyrusDataset:
        """Create a :class:`PapyrusDataset` from raw components.

        Used internally by filter methods to propagate the current state.

        :param papyrus_bioactivity_data: (filtered) bioactivity data
        :param papyrus_protein_data: protein-target DataFrame
        :param papyrus_params: parameters dict from the parent dataset
        """
        dataset = PapyrusDataset.__new__(PapyrusDataset)
        dataset.papyrus_bioactivity_data = papyrus_bioactivity_data
        dataset.papyrus_protein_data = papyrus_protein_data
        dataset.papyrus_params = papyrus_params
        dataset._fpsubsim2_ = None
        dataset._can_reset = False
        return dataset

    # ------------------------------------------------------------------
    # Internal properties
    # ------------------------------------------------------------------

    @property
    def _filter(self) -> PapyrusDataFilter:
        """Return a :class:`PapyrusDataFilter` wrapping the current state."""
        return PapyrusDataFilter(
            papyrus_bioactivity_data=self.papyrus_bioactivity_data,
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params,
        )

    @property
    def _fpsubsim2(self) -> FPSubSim2Engine:
        """Return (and lazily create) the :class:`FPSubSim2Engine` for this dataset."""
        if self._fpsubsim2_ is None:
            self._fpsubsim2_ = FPSubSim2Engine(self.papyrus_params)
        self._fpsubsim2_._set_data(
            papyrus_bioactivity_data=self.papyrus_bioactivity_data,
            papyrus_protein_data=self.papyrus_protein_data,
        )
        return self._fpsubsim2_

    # ------------------------------------------------------------------
    # Filters — bioactivity
    # ------------------------------------------------------------------

    def keep_quality(self, min_quality: str) -> PapyrusDataset:
        """Keep samples at or above the given quality level.

        :param min_quality: minimum quality: ``'low'``, ``'medium'``, or
            ``'high'``
        """
        return self._filter.keep_quality(min_quality=min_quality)

    def keep_source(self, source: list[str] | str) -> PapyrusDataset:
        """Keep samples from specific data source(s).

        :param source: source label(s) such as ``'chembl'`` or
            ``['chembl', 'klaeger']``
        """
        return self._filter.keep_source(source=source)

    def keep_activity_type(self, activity_types: list[str] | str) -> PapyrusDataset:
        """Keep samples of specific activity type(s).

        :param activity_types: type(s) such as ``'ic50'`` or
            ``['ki', 'ec50']``
        """
        return self._filter.keep_activity_type(activity_types=activity_types)

    def keep_accession(self, accession: list[str] | str = 'all') -> PapyrusDataset:
        """Keep samples matching the given UniProt accession(s).

        :param accession: accession code(s) such as ``'P00533'`` or
            ``['P11362', 'P35968']``
        """
        return self._filter.keep_accession(accession=accession)

    # ------------------------------------------------------------------
    # Filters — protein
    # ------------------------------------------------------------------

    def keep_protein_class(
            self,
            classes: dict | list[dict] | None,
            generic_regex: bool = False,
    ) -> PapyrusDataset:
        """Keep samples whose targets belong to the given protein class(es).

        :param classes: protein class filter — see
            :func:`~preprocess.keep_protein_class` for the full syntax
        :param generic_regex: treat ``'l?'`` patterns as regular expressions
        """
        return self._filter.keep_protein_class(classes=classes, generic_regex=generic_regex)

    def keep_organism(
            self,
            organism: str | list[str] | None,
            generic_regex: bool = False,
    ) -> PapyrusDataset:
        """Keep samples whose targets come from the specified organism(s).

        :param organism: organism name(s) such as
            ``'Homo sapiens (Human)'``
        :param generic_regex: allow partial / regex matching of organism names
        """
        return self._filter.keep_organism(organism=organism, generic_regex=generic_regex)

    # ------------------------------------------------------------------
    # Filters — generic column
    # ------------------------------------------------------------------

    def contains(
            self,
            column: str,
            value: str,
            case: bool = True,
            regex: bool = False,
    ) -> PapyrusDataset:
        """Keep samples whose *column* contains *value*.

        :param column: column to inspect
        :param value: substring or pattern to match
        :param case: case-sensitive match
        :param regex: interpret *value* as a regular expression
        """
        return self._filter.contains(column=column, value=value, case=case, regex=regex)

    def not_contains(
            self,
            column: str,
            value: str,
            case: bool = True,
            regex: bool = False,
    ) -> PapyrusDataset:
        """Keep samples whose *column* does **not** contain *value*.

        :param column: column to inspect
        :param value: substring or pattern to exclude
        :param case: case-sensitive match
        :param regex: interpret *value* as a regular expression
        """
        return self._filter.not_contains(column=column, value=value, case=case, regex=regex)

    def isin(self, column: str, values: Any | list[Any]) -> PapyrusDataset:
        """Keep samples whose *column* value is in *values*.

        :param column: column to inspect
        :param values: acceptable value(s)
        """
        return self._filter.isin(column=column, values=values)

    def not_isin(self, column: str, values: Any | list[Any]) -> PapyrusDataset:
        """Keep samples whose *column* value is **not** in *values*.

        :param column: column to inspect
        :param values: values to exclude
        """
        return self._filter.not_isin(column=column, values=values)

    # ------------------------------------------------------------------
    # Filters — molecular structure
    # ------------------------------------------------------------------

    def keep_similar_molecules(
            self,
            smiles: str | list[str],
            fp: Fingerprint | None = None,
            threshold: float = 0.7,
            cuda: bool = False,
    ) -> PapyrusDataset:
        """Keep samples with structures similar to any of the given SMILES.

        :param smiles: query SMILES string(s)
        :param fp: fingerprint type to use (default: :class:`~fingerprint.MorganFingerprint`)
        :param threshold: Tanimoto similarity threshold (default: 0.7)
        :param cuda: use GPU-accelerated search
        """
        if fp is None:
            fp = MorganFingerprint()
        return self._fpsubsim2.keep_similar_molecules(smiles=smiles, fp=fp,
                                                      threshold=threshold, cuda=cuda,
                                                      )

    def keep_dissimilar_molecules(
            self,
            smiles: str | list[str],
            fp: Fingerprint | None = None,
            threshold: float = 0.7,
            cuda: bool = False,
    ) -> PapyrusDataset:
        """Keep samples with structures **not** similar to any of the given SMILES.

        :param smiles: query SMILES string(s)
        :param fp: fingerprint type to use (default: :class:`~fingerprint.MorganFingerprint`)
        :param threshold: Tanimoto similarity threshold (default: 0.7)
        :param cuda: use GPU-accelerated search
        """
        if fp is None:
            fp = MorganFingerprint()
        return self._fpsubsim2.keep_dissimilar_molecules(smiles=smiles, fp=fp,
                                                         threshold=threshold, cuda=cuda,
                                                         )

    def keep_substructure_molecules(self, smiles: str | list[str]) -> PapyrusDataset:
        """Keep samples whose structures contain any of the given SMILES as a substructure.

        :param smiles: query SMILES string(s)
        """
        return self._fpsubsim2.keep_substructure_molecules(smiles=smiles)

    def keep_not_substructure_molecules(self, smiles: str | list[str]) -> PapyrusDataset:
        """Keep samples whose structures do **not** contain any of the given SMILES as a substructure.

        :param smiles: query SMILES string(s)
        """
        return self._fpsubsim2.keep_not_substructure_molecules(smiles=smiles)

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def _num_rows(self) -> int | None:
        """Return the total bioactivity row count - resolving (downloading first if needed) it if unknown."""
        source = self.papyrus_params.get('_source')
        if source is not None:
            return source.get_num_rows()
        return self.papyrus_params.get('num_rows')

    def aggregate(self, progress: bool = False) -> pl.DataFrame:
        """Materialise all lazy filters into a single :class:`~polars.DataFrame`.

        Downloads the bioactivity file first if not yet available locally.

        :param progress: show a tqdm progress bar while consuming chunks
        :returns: a DataFrame of the filtered data
        """
        data = self.papyrus_bioactivity_data
        if isinstance(data, _LazyBioactivity):
            data = data.resolve()
        if isinstance(data, pl.DataFrame):
            return data
        total = _num_chunks(self._num_rows(), self.papyrus_params['chunksize'])
        return preprocess.consume_chunks(
            generator=data, progress=progress, total=total,
        )

    #: Alias for :meth:`aggregate`.
    def agg(self, progress: bool = False) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def consume_chunks(self, progress: bool = False) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def to_dataframe(self, progress: bool = False) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    # ------------------------------------------------------------------
    # Derived datasets
    # ------------------------------------------------------------------

    def molecules(
            self,
            chunksize: int | None = 1_000_000,
            progress: bool = False,
    ) -> PapyrusMoleculeSet:
        """Return the molecular structures for the samples in this dataset.

        Purely lazy: neither this dataset's bioactivity data nor the
        structure file itself are touched here - both happen only once
        :meth:`PapyrusMoleculeSet.aggregate` (or an alias) is called on the
        returned object, at which point the structure file is downloaded
        first if not yet available locally.

        :param chunksize: structures per chunk (default: 1 000 000)
        :param progress: default ``progress`` for the returned set's
            :meth:`~PapyrusMoleculeSet.aggregate` when not overridden there
        :returns: a :class:`PapyrusMoleculeSet`
        """
        source = self.papyrus_params.get('_source')
        if source is not None:
            source.request_structures()
        return PapyrusMoleculeSet(self, chunksize=chunksize, progress=progress)

    def proteins(self, progress: bool = False) -> PapyrusProteinSet:
        """Return the protein targets for the samples in this dataset.

        :param progress: show progress while aggregating the bioactivity data
        :returns: a :class:`PapyrusProteinSet`
        """
        ids = self.aggregate(progress=progress)['target_id'].unique()
        protein_data = _resolve(self.papyrus_protein_data)
        proteins = protein_data.filter(pl.col('target_id').is_in(ids.implode()))
        return PapyrusProteinSet(proteins, self.papyrus_params, num_proteins=len(proteins))

    def match_rcsb_pdb(self, update: bool = True, progress: bool = False) -> PapyrusPDBProteinSet:
        """Match samples to RCSB Protein Data Bank 3D structures.

        Downloads the bioactivity file first if not yet available locally.

        :param update: refresh the local PDB identifier cache (default: True)
        :param progress: show progress while matching
        :returns: a :class:`PapyrusPDBProteinSet`
        """
        data = self.papyrus_bioactivity_data
        if isinstance(data, _LazyBioactivity):
            data = data.resolve()
        total = _num_chunks(self._num_rows(), self.papyrus_params['chunksize'])
        structures = get_pdb_matches(
            data,
            root_folder=self.papyrus_params['source_path'],
            verbose=progress,
            total=total,
            update=update,
        )
        return PapyrusPDBProteinSet(structures, self.papyrus_params)

    # ------------------------------------------------------------------
    # Descriptors
    # ------------------------------------------------------------------

    def molecular_descriptors(
            self,
            desc_type: str,
            progress: bool = False,
    ) -> PapyrusDescriptorSet:
        """Return molecular descriptors for the molecules in this dataset.

        Purely lazy: neither this dataset's bioactivity data nor the
        descriptor file itself are touched here - both happen only once
        :meth:`PapyrusDescriptorSet.aggregate` (or an alias) is called on
        the returned object, at which point the descriptor file is
        downloaded first if not yet available locally.

        :param desc_type: descriptor set; one of ``'mold2'``, ``'mordred'``,
            ``'cddd'``, ``'fingerprint'``, ``'moe'``, ``'all'``
        :param progress: default ``progress`` for the returned set's
            :meth:`~PapyrusDescriptorSet.aggregate` when not overridden there
        :returns: a :class:`PapyrusDescriptorSet`
        """
        source = self.papyrus_params.get('_source')
        if source is not None:
            source.request_descriptors(desc_type)
        return PapyrusDescriptorSet(self, desc_type=desc_type, progress=progress)

    # ------------------------------------------------------------------
    # Administration
    # ------------------------------------------------------------------

    def reset(self) -> bool:
        """Reset the underlying data stream to the beginning of the file.

        Has no effect when the dataset was created from a DataFrame. Does
        not by itself download anything not already fetched - it reuses
        this dataset's :class:`_PapyrusSource` (downloading only if that
        hasn't happened yet either).

        :returns: ``True`` if the stream was reset, ``False`` otherwise
        """
        if self._can_reset:
            source = self.papyrus_params['_source']
            self.papyrus_bioactivity_data = _LazyBioactivity(loader=source.get_bioactivity)
            self.papyrus_protein_data = _Deferred(loader=source.get_proteins)
        return self._can_reset

    @staticmethod
    def remove(
            version: str | IO.PapyrusVersion,
            remove_papyruspp: bool,
            remove_bioactivities: bool,
            remove_proteins: bool,
            remove_nostereo: bool,
            remove_stereo: bool,
            remove_structures: bool,
            remove_descriptors: str | list[str],
            remove_other_files: bool,
            remove_version_root: bool,
            remove_papyrus_root: bool,
            force: bool = False,
            progress: bool = True,
            source_path: str | Path | None = None,
            all_revisions: bool = False,
    ) -> None:
        """Remove locally downloaded Papyrus data.

        All arguments map directly to :func:`~download.remove_papyrus`; see
        that function for full documentation.
        """
        # 'latest' and 'all' are sentinels handled by remove_papyrus directly.
        if isinstance(version, str) and version in ('latest', 'all'):
            version_arg = version
        else:
            pv = _ensure_papyrus_version(version)
            # pystow_path_key (not .version) so remove_papyrus resolves the
            # same on-disk folder download_papyrus wrote to - .version is the
            # canonical new-format string, which for an old-format version
            # would make pystow.module() create an empty sibling folder
            # under the wrong (new-format) name instead of touching the
            # real, old-format one.
            version_arg = pv.pystow_path_key
        download.remove_papyrus(
            outdir=source_path,
            version=version_arg,
            papyruspp=remove_papyruspp,
            bioactivities=remove_bioactivities,
            proteins=remove_proteins,
            nostereo=remove_nostereo,
            stereo=remove_stereo,
            structures=remove_structures,
            descriptors=remove_descriptors,
            other_files=remove_other_files,
            version_root=remove_version_root,
            papyrus_root=remove_papyrus_root,
            force=force,
            progress=progress,
            all_revisions=all_revisions,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Show the class name and its non-internal papyrus_params."""
        params = ', '.join(
            f'{k}={v}' for k, v in self.papyrus_params.items() if not k.startswith('_')
        )
        return f'{type(self).__name__}<{params}>'


# ---------------------------------------------------------------------------
# Filter-method generation
# ---------------------------------------------------------------------------
#
# PapyrusDataFilter and FPSubSim2Engine each wrap a handful of preprocess.py
# filter functions with a thin "supply data/context, call preprocess.*, wrap
# the result back into a PapyrusDataset" body. Writing those bodies out by
# hand duplicates every preprocess.py parameter list a second time, with no
# signal if the two drift apart. _FilterSpec/_build_filter_method/
# _generate_filters generate that body from preprocess.py's own signature
# instead, so a new parameter on a preprocess.* function is picked up here
# automatically.

@dataclass(frozen=True)
class _FilterSpec:
    """Declarative description of one generated filter method.

    Only ``target_name`` is required. Every other field defaults to "no
    special handling": a pure passthrough of the preprocess.py signature,
    with ``data`` always supplied from ``self.papyrus_bioactivity_data``.
    A preprocess.py parameter added later without a matching entry here is
    auto-exposed on the generated method - use ``context`` to keep an
    internal-only parameter off the public surface instead.
    """

    #: name of the function in preprocess.py to delegate to
    target_name: str
    #: method name on the composition class, if it differs from target_name
    name: str | None = None
    #: {local (method-facing) param name: preprocess.py param name}, for
    #: params whose public name differs between the two layers
    renames: Mapping[str, str] = field(default_factory=dict)
    #: {preprocess.py param name: attribute name on self} - params supplied
    #: from instance state rather than by the caller (never publicly exposed)
    context: Mapping[str, str] = field(default_factory=dict)
    #: {local param name: zero-arg factory}, for params whose *public*
    #: default must be None (avoiding a mutable/instance default in the
    #: generated signature) and are resolved to a real value via factory()
    #: only when the caller passes/leaves None
    optional_defaults: Mapping[str, Callable[[], Any]] = field(default_factory=dict)
    #: local param names whose preprocess.py default must be dropped (forced
    #: required), because the OOP layer intentionally requires them
    #: explicitly even though preprocess.py defaults them
    required: frozenset = field(default_factory=frozenset)
    #: optional callable run on self before delegating, e.g.
    #: FPSubSim2Engine._ensure_loaded
    pre_hook: Callable[[Any], None] | None = None


def _build_filter_method(spec: _FilterSpec) -> Callable:
    """Build a method delegating to ``preprocess.<target_name>`` with a real signature."""
    target = getattr(preprocess, spec.target_name)
    sig = inspect.signature(target)
    excluded = {'data', *spec.context}

    params = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for pname, p in sig.parameters.items():
        if pname in excluded:
            continue
        local_name = next((k for k, v in spec.renames.items() if v == pname), pname)
        annotation, default = p.annotation, p.default
        if local_name in spec.optional_defaults:
            default = None
            if annotation is not inspect.Parameter.empty:
                annotation = annotation | None
        elif local_name in spec.required:
            default = inspect.Parameter.empty
        params.append(p.replace(name=local_name, default=default, annotation=annotation))

    new_sig = inspect.Signature(params, return_annotation='PapyrusDataset')

    def method(self, *args, **kwargs):
        bound = new_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if spec.pre_hook is not None:
            spec.pre_hook(self)
        call_kwargs = {tname: getattr(self, attr) for tname, attr in spec.context.items()}
        for local_name, value in bound.arguments.items():
            if local_name == 'self':
                continue
            if local_name in spec.optional_defaults and value is None:
                value = spec.optional_defaults[local_name]()
            call_kwargs[spec.renames.get(local_name, local_name)] = value
        # Re-resolve preprocess.<target_name> at call time (not the `target`
        # captured above) so unittest.mock.patch on preprocess.* still works.
        # Routed through _apply_lazy so a not-yet-downloaded (_LazyBioactivity)
        # dataset stays lazy instead of eagerly resolving here.
        target = getattr(preprocess, spec.target_name)
        return self._wrap(_apply_lazy(self.papyrus_bioactivity_data, target, **call_kwargs))

    method.__name__ = method.__qualname__ = spec.name or spec.target_name
    method.__signature__ = new_sig
    summary = (inspect.getdoc(target) or '').split('\n', 1)[0]
    method.__doc__ = f'{summary}\n\nSee :func:`~preprocess.{spec.target_name}` for full parameter documentation.'
    method.__module__ = __name__
    return method


def _generate_filters(specs: list[_FilterSpec]) -> Callable[[type], type]:
    """Class decorator attaching one generated method per spec in *specs*."""
    def decorator(cls: type) -> type:
        for spec in specs:
            m = _build_filter_method(spec)
            setattr(cls, m.__name__, m)
        return cls
    return decorator


# ---------------------------------------------------------------------------
# PapyrusDataFilter
# ---------------------------------------------------------------------------

_PAPYRUS_DATA_FILTER_SPECS = [
    _FilterSpec(target_name='keep_quality'),
    _FilterSpec(target_name='keep_source'),
    _FilterSpec(target_name='keep_type', name='keep_activity_type'),
    _FilterSpec(target_name='keep_accession'),
    _FilterSpec(
        target_name='keep_protein_class',
        context={'protein_data': 'papyrus_protein_data'},
        required=frozenset({'classes'}),
    ),
    _FilterSpec(
        target_name='keep_organism',
        context={'protein_data': 'papyrus_protein_data'},
    ),
    _FilterSpec(target_name='keep_contains', name='contains'),
    _FilterSpec(target_name='keep_not_contains', name='not_contains'),
    _FilterSpec(target_name='keep_match', name='isin'),
    _FilterSpec(target_name='keep_not_match', name='not_isin'),
]


@_generate_filters(_PAPYRUS_DATA_FILTER_SPECS)
class PapyrusDataFilter:
    """Collection of filters applied to a :class:`PapyrusDataset`.

    Normally obtained via :attr:`PapyrusDataset._filter`.  Can be configured
    for parallelism and verbosity via :meth:`__call__` before chaining a
    filter method::

        dataset._filter(njobs=4, progress=True).keep_quality('medium')

    Filter methods (``keep_quality``, ``keep_source``, ``keep_activity_type``,
    ``keep_accession``, ``keep_protein_class``, ``keep_organism``,
    ``contains``, ``not_contains``, ``isin``, ``not_isin``) are generated from
    :mod:`preprocess`'s function signatures - see ``_PAPYRUS_DATA_FILTER_SPECS``.
    """

    def __init__(
            self,
            papyrus_bioactivity_data: Iterator[pd.DataFrame] | pd.DataFrame,
            papyrus_protein_data: pd.DataFrame,
            papyrus_params: dict,
            njobs: int = 1,
            progress: bool = False,
    ) -> None:
        """Wrap bioactivity/protein data and params for the generated filter methods to use."""
        self.papyrus_bioactivity_data = papyrus_bioactivity_data
        self.papyrus_protein_data = papyrus_protein_data
        self.papyrus_params = papyrus_params
        self.njobs = njobs
        self.progress = progress

    def __call__(self, njobs: int = 1, progress: bool = False) -> PapyrusDataFilter:
        """Configure parallelism and verbosity, then return *self* for chaining.

        :param njobs: number of parallel jobs for aggregation-heavy filters
        :param progress: show swifter / tqdm progress bars
        :returns: *self* (enables method chaining)
        """
        self.njobs = njobs
        self.progress = progress
        return self

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _wrap(self, filtered_data) -> PapyrusDataset:
        """Wrap *filtered_data* in a new :class:`PapyrusDataset`."""
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=filtered_data,
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params,
        )

    # Filter methods (keep_quality, keep_source, keep_activity_type,
    # keep_accession, keep_protein_class, keep_organism, contains,
    # not_contains, isin, not_isin) are attached by @_generate_filters above.
    # Declarations below are for IDE autocomplete/mypy only - never executed,
    # so they don't conflict with the real methods @_generate_filters
    # attaches. Regenerate via scripts/generate_oop_filter_stubs.py if a
    # preprocess.py filter's signature changes.
    if TYPE_CHECKING:
        def keep_quality(self, min_quality: str = 'high') -> PapyrusDataset:
            """Keep only rows at or above the minimum required quality level."""
            ...

        def keep_source(self, source: list[str] | str = 'all', validate: bool = True) -> PapyrusDataset:
            """Keep only rows from the specified data source(s)."""
            ...

        def keep_activity_type(
                self, activity_types: list[str] | str = 'ic50', validate: bool = True,
        ) -> PapyrusDataset:
            """Keep only rows matching the desired activity type(s)."""
            ...

        def keep_accession(self, accession: list[str] | str = 'all') -> PapyrusDataset:
            """Keep only rows whose target ID matches the given UniProt accession(s)."""
            ...

        def keep_protein_class(
                self, classes: dict | list[dict] | None, generic_regex: bool = False,
        ) -> PapyrusDataset:
            """Keep only rows whose target belongs to the desired protein class(es)."""
            ...

        def keep_organism(
                self, organism: str | list[str] | None = 'Homo sapiens (Human)', generic_regex: bool = False,
        ) -> PapyrusDataset:
            """Keep only rows whose target comes from the specified organism(s)."""
            ...

        def contains(self, column: str, value: str, case: bool = True, regex: bool = False) -> PapyrusDataset:
            """Keep only rows where *column* contains *value*."""
            ...

        def not_contains(self, column: str, value: str, case: bool = True, regex: bool = False) -> PapyrusDataset:
            """Keep only rows where *column* does **not** contain *value*."""
            ...

        def isin(self, column: str, values: Any | list[Any]) -> PapyrusDataset:
            """Keep only rows where *column* is in *values* (equivalent to ``is_in``)."""
            ...

        def not_isin(self, column: str, values: Any | list[Any]) -> PapyrusDataset:
            """Keep only rows where *column* is **not** in *values*."""
            ...


# ---------------------------------------------------------------------------
# FPSubSim2Engine
# ---------------------------------------------------------------------------

_FPSUBSIM2_SPECS = [
    _FilterSpec(
        target_name='keep_similar', name='keep_similar_molecules',
        renames={'smiles': 'molecule_smiles', 'fp': 'fingerprint'},
        context={'fpsubsim2_file': 'path'},
        optional_defaults={'fp': lambda: MorganFingerprint()},
        pre_hook=lambda self: self._ensure_loaded(),
    ),
    _FilterSpec(
        target_name='keep_dissimilar', name='keep_dissimilar_molecules',
        renames={'smiles': 'molecule_smiles', 'fp': 'fingerprint'},
        context={'fpsubsim2_file': 'path'},
        optional_defaults={'fp': lambda: MorganFingerprint()},
        pre_hook=lambda self: self._ensure_loaded(),
    ),
    _FilterSpec(
        target_name='keep_substructure', name='keep_substructure_molecules',
        renames={'smiles': 'molecule_smiles'},
        context={'fpsubsim2_file': 'path'},
        pre_hook=lambda self: self._ensure_loaded(),
    ),
    _FilterSpec(
        target_name='keep_not_substructure', name='keep_not_substructure_molecules',
        renames={'smiles': 'molecule_smiles'},
        context={'fpsubsim2_file': 'path'},
        pre_hook=lambda self: self._ensure_loaded(),
    ),
]


@_generate_filters(_FPSUBSIM2_SPECS)
class FPSubSim2Engine:
    """Manages an FPSubSim2 similarity/substructure search database for a :class:`PapyrusDataset`.

    Filter methods (``keep_similar_molecules``, ``keep_dissimilar_molecules``,
    ``keep_substructure_molecules``, ``keep_not_substructure_molecules``) are
    generated from :mod:`preprocess`'s function signatures - see
    ``_FPSUBSIM2_SPECS``.
    """

    def __init__(self, papyrus_params: dict) -> None:
        """Store *papyrus_params*; the FPSubSim2 database itself is loaded lazily."""
        self.papyrus_params = papyrus_params
        self.path: str | Path | None = None
        self.progress: bool = False
        self.fp: Fingerprint | list[Fingerprint] = MorganFingerprint()
        self.fpsubsim2 = subsim_search.FPSubSim2()
        self.papyrus_bioactivity_data: Iterator[pd.DataFrame] | pd.DataFrame | None = None
        self.papyrus_protein_data: pd.DataFrame | None = None

    def __call__(
            self,
            fp: Fingerprint | list[Fingerprint] | None = None,
            path: str | Path | None = None,
            progress: bool = False,
    ) -> FPSubSim2Engine:
        """Configure the engine and return *self* for chaining.

        :param fp: fingerprint(s) to use when creating a new database
            (default: :class:`~fingerprint.MorganFingerprint`)
        :param path: explicit path for the ``.h5`` file; auto-derived when
            ``None``
        :param progress: show progress bars during database creation
        :returns: *self*
        """
        self.fp = fp if fp is not None else MorganFingerprint()
        self.path = path
        self.progress = progress
        return self

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_path(self) -> Path:
        """Derive the default ``.h5`` path from ``papyrus_params``.

        :returns: absolute path to the FPSubSim2 database file
        :raises NotADirectoryError: if the parent directory does not exist
        """
        pv = self.papyrus_params['version']  # PapyrusVersion
        is3d = self.papyrus_params['is3d']
        dim_tag = '3D' if is3d else '2D'
        stereo = 'without' if not is3d else 'with'
        name = (f'{pv.pystow_path_key}_combined_set_'
                f'{stereo}_stereochemistry_FPSubSim2_{dim_tag}.h5')

        IO._set_root_folder(self.papyrus_params['source_path'])

        path = pystow.module('papyrus', pv.pystow_path_key).join(name=name)

        parent = path.parent
        if not parent.is_dir():
            raise NotADirectoryError(
                f'Cannot create the FPSubSim2 file in a non-existing folder: {parent!r}',
            )
        return path

    def _ensure_loaded(self) -> None:
        """Load or create the FPSubSim2 database as needed."""
        if self.path is None:
            self.path = self._resolve_path()

        if Path(self.path).is_file():
            self.fpsubsim2.load(fpsubsim_path=self.path)
        else:
            self.fpsubsim2.create_from_papyrus(
                is3d=self.papyrus_params['is3d'],
                version=self.papyrus_params['version'],
                outfile=self.path,
                fingerprint=self.fp,
                root_folder=self.papyrus_params['source_path'],
                progress=self.progress,
            )

    def _set_data(
            self,
            papyrus_bioactivity_data: Iterator[pd.DataFrame] | pd.DataFrame,
            papyrus_protein_data: pd.DataFrame,
    ) -> None:
        """Attach current bioactivity and protein data to this engine."""
        self.papyrus_bioactivity_data = papyrus_bioactivity_data
        self.papyrus_protein_data = papyrus_protein_data

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _wrap(self, filtered_data) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=filtered_data,
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params,
        )

    # Filter methods (keep_similar_molecules, keep_dissimilar_molecules,
    # keep_substructure_molecules, keep_not_substructure_molecules) are
    # attached by @_generate_filters above. See PapyrusDataFilter above for
    # why the TYPE_CHECKING declarations below exist.
    if TYPE_CHECKING:
        def keep_similar_molecules(
                self, smiles: str | list[str], fp: Fingerprint | None = None,
                threshold: float = 0.7, cuda: bool = False,
        ) -> PapyrusDataset:
            """Keep only rows associated to molecules similar to the query."""
            ...

        def keep_dissimilar_molecules(
                self, smiles: str | list[str], fp: Fingerprint | None = None,
                threshold: float = 0.7, cuda: bool = False,
        ) -> PapyrusDataset:
            """Keep only rows associated to molecules **not** similar to the query."""
            ...

        def keep_substructure_molecules(self, smiles: str | list[str]) -> PapyrusDataset:
            """Keep only rows associated to substructures of the query molecule(s)."""
            ...

        def keep_not_substructure_molecules(self, smiles: str | list[str]) -> PapyrusDataset:
            """Keep only rows associated to molecules that are **not** substructures of the query."""
            ...


# ---------------------------------------------------------------------------
# PapyrusMoleculeSet
# ---------------------------------------------------------------------------

class PapyrusMoleculeSet:
    """A set of molecular structures derived from a :class:`PapyrusDataset`.

    Purely lazy: constructing this class (via :meth:`PapyrusDataset.molecules`)
    neither materialises the parent dataset's bioactivity data nor touches
    the structure file - both happen only once :meth:`aggregate` (or an
    alias) is called, at which point the structure file is downloaded first
    if not yet available locally.
    """

    def __init__(
            self,
            dataset: PapyrusDataset,
            chunksize: int | None = 1_000_000,
            progress: bool = False,
    ) -> None:
        """Record *dataset* and *chunksize*; nothing is loaded until :meth:`aggregate`."""
        self._dataset = dataset
        self._default_progress = progress
        self.papyrus_params: dict = {**dataset.papyrus_params, 'chunksize': chunksize}
        self.data: pl.DataFrame | Iterator | None = None
        self.num_rows: int | None = None

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(self, progress: bool) -> None:
        """Download the structure file (if needed) and read it - once."""
        if self.data is not None:
            return
        ids = self._dataset.aggregate(progress=progress)[_id_column(self.papyrus_params['is3d'])].unique()
        read_kwargs = dict(
            is3d=self.papyrus_params['is3d'],
            version=self.papyrus_params['version'],
            chunksize=self.papyrus_params['chunksize'],
            source_path=self.papyrus_params['source_path'],
            ids=ids, verbose=False,
        )
        try:
            self.data = reader.read_molecular_structures(**read_kwargs)
        except _NOT_AVAILABLE_LOCALLY:
            download.download_papyrus(
                outdir=self.papyrus_params['source_path'],
                # pystow_path_key (not .version): must match the folder key
                # every read in this class looks up self.papyrus_params['version']
                # under, or an old-format pv silently downloads to the wrong folder.
                version=self.papyrus_params['version'].pystow_path_key,
                nostereo=not self.papyrus_params['is3d'],
                stereo=self.papyrus_params['is3d'],
                only_pp=False,
                structures=True,
                descriptors=None,
                progress=self.papyrus_params['download_progress'],
                disk_margin=self.papyrus_params['disk_margin'],
                keep_xz=self.papyrus_params['keep_original_files'],
            )
            self.data = reader.read_molecular_structures(**read_kwargs)
        self.num_rows = IO.get_num_rows_in_file(
            filetype='structures',
            is3D=self.papyrus_params['is3d'],
            version=self.papyrus_params['version'],
            root_folder=self.papyrus_params['source_path'],
        )

    def aggregate(self, progress: bool | None = None) -> pl.DataFrame:
        """Materialise all structures into a single :class:`~polars.DataFrame`.

        Downloads the structure file first if not yet available locally.

        :param progress: show a progress bar; defaults to the value passed
            to :meth:`PapyrusDataset.molecules`
        """
        progress = self._default_progress if progress is None else progress
        self._ensure_loaded(progress)
        if isinstance(self.data, pl.DataFrame):
            return self.data
        if self.data is None:
            raise RuntimeError('structures not loaded despite _ensure_loaded()')
        total = _num_chunks(self.num_rows, self.papyrus_params['chunksize'])
        return preprocess.consume_chunks(generator=self.data, progress=progress, total=total)

    def agg(self, progress: bool | None = None) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def consume_chunks(self, progress: bool | None = None) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def to_dataframe(self, progress: bool | None = None) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    # ------------------------------------------------------------------
    # Descriptors
    # ------------------------------------------------------------------

    def molecular_descriptors(
            self,
            desc_type: str,
            progress: bool = False,
    ) -> PapyrusDescriptorSet:
        """Return molecular descriptors for the molecules in this set.

        Purely lazy: see :meth:`PapyrusDataset.molecular_descriptors`.

        :param desc_type: one of ``'mold2'``, ``'mordred'``, ``'cddd'``,
            ``'fingerprint'``, ``'moe'``, ``'all'``
        :param progress: default ``progress`` for the returned set's
            :meth:`~PapyrusDescriptorSet.aggregate` when not overridden there
        :returns: a :class:`PapyrusDescriptorSet`
        """
        source = self.papyrus_params.get('_source')
        if source is not None:
            source.request_descriptors(desc_type)
        return PapyrusDescriptorSet(self, desc_type=desc_type, progress=progress)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Show materialisation state, and the molecule count once materialised."""
        if self.data is None:
            return f'{type(self).__name__}<not yet materialised>'
        if not isinstance(self.data, pl.DataFrame):
            return f'{type(self).__name__}<iterator of molecules>'
        return f'{type(self).__name__}<{len(self.data)} molecules>'


# ---------------------------------------------------------------------------
# PapyrusDescriptorSet
# ---------------------------------------------------------------------------

class PapyrusDescriptorSet:
    """A set of molecular descriptors derived from a :class:`PapyrusDataset` or :class:`PapyrusMoleculeSet`.

    Purely lazy: constructing this class (via
    :meth:`PapyrusDataset.molecular_descriptors` or
    :meth:`PapyrusMoleculeSet.molecular_descriptors`) neither materialises
    the parent set nor touches the descriptor file - both happen only once
    :meth:`aggregate` (or an alias) is called, at which point the
    descriptor file is downloaded first if not yet available locally.
    """

    def __init__(
            self,
            dataset: PapyrusDataset | PapyrusMoleculeSet,
            desc_type: str,
            progress: bool = False,
    ) -> None:
        """Record *dataset* and *desc_type*; nothing is loaded until :meth:`aggregate`."""
        self._dataset = dataset
        self._desc_type = desc_type
        self._default_progress = progress
        self.papyrus_params: dict = dict(dataset.papyrus_params)
        self.data: pl.DataFrame | pl.LazyFrame | None = None

    def _ensure_loaded(self, progress: bool) -> None:
        """Download the descriptor file (if needed) and read it - once."""
        if self.data is not None:
            return
        ids = self._dataset.aggregate(progress=progress)[_id_column(self.papyrus_params['is3d'])].unique()
        read_kwargs = dict(
            desc_type=self._desc_type,
            is3d=self.papyrus_params['is3d'],
            version=self.papyrus_params['version'],
            chunksize=self.papyrus_params['chunksize'],
            source_path=self.papyrus_params['source_path'],
            ids=ids,
            verbose=progress,
        )
        try:
            self.data = reader.read_molecular_descriptors(**read_kwargs)
        except _NOT_AVAILABLE_LOCALLY:
            download.download_papyrus(
                outdir=self.papyrus_params['source_path'],
                # pystow_path_key (not .version): must match the folder key
                # every read in this class looks up self.papyrus_params['version']
                # under, or an old-format pv silently downloads to the wrong folder.
                version=self.papyrus_params['version'].pystow_path_key,
                nostereo=not self.papyrus_params['is3d'],
                stereo=self.papyrus_params['is3d'],
                only_pp=self.papyrus_params['plusplus'],
                structures=False,
                descriptors=self._desc_type,
                progress=self.papyrus_params['download_progress'],
                disk_margin=self.papyrus_params['disk_margin'],
                keep_xz=self.papyrus_params['keep_original_files'],
            )
            self.data = reader.read_molecular_descriptors(**read_kwargs)

    def aggregate(self, progress: bool | None = None) -> pl.DataFrame:
        """Materialise all descriptors into a single :class:`~polars.DataFrame`.

        Downloads the descriptor file first if not yet available locally.

        :param progress: show a progress bar; defaults to the value passed
            to :meth:`PapyrusDataset.molecular_descriptors`
        """
        progress = self._default_progress if progress is None else progress
        self._ensure_loaded(progress)
        if self.data is None:
            raise RuntimeError('descriptors not loaded despite _ensure_loaded()')
        return preprocess.consume_chunks(generator=self.data, progress=progress, total=None)

    def agg(self, progress: bool | None = None) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def consume_chunks(self, progress: bool | None = None) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def to_dataframe(self, progress: bool | None = None) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Show the descriptor type and materialisation state."""
        if self.data is None:
            return f'{type(self).__name__}<{self._desc_type}, not yet materialised>'
        return f'{type(self).__name__}<{self._desc_type}>'


# ---------------------------------------------------------------------------
# ProteinSet  (abstract base)
# ---------------------------------------------------------------------------

class ProteinSet(ABC):
    """Abstract base for protein-target set classes."""

    data: pd.DataFrame | pl.DataFrame
    papyrus_params: dict

    @abstractmethod
    def aggregate(self, progress: bool = False) -> pd.DataFrame | pl.DataFrame:
        """Materialise the protein data into a DataFrame."""

    def protein_descriptors(
            self,
            desc_type: str | prodec.Descriptor | prodec.Transform,
            progress: bool = False,
            custom_descriptor_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Return protein descriptors for the targets in this set.

        Downloads the descriptor file if not yet available locally (except
        for ``'custom'`` - not Papyrus-hosted).

        :param desc_type: descriptor type: ``'unirep'``, ``'custom'``, or a
            ProDEC :class:`~prodec.Descriptor` / :class:`~prodec.Transform`
        :param progress: show progress while aggregating
        :param custom_descriptor_path: path to the custom TSV descriptor
            file; required (and only used) when *desc_type* is ``'custom'``
        """
        self.data = self.aggregate(progress)
        ids = self.data['target_id'].unique()
        if desc_type == 'custom':
            return reader.read_protein_descriptors(
                desc_type=desc_type,
                version=self.papyrus_params['version'],
                chunksize=self.papyrus_params['chunksize'],
                source_path=custom_descriptor_path,
                ids=ids,
                verbose=progress,
            )
        try:
            return reader.read_protein_descriptors(
                desc_type=desc_type,
                version=self.papyrus_params['version'],
                chunksize=self.papyrus_params['chunksize'],
                source_path=self.papyrus_params['source_path'],
                ids=ids,
                verbose=progress,
            )
        except _NOT_AVAILABLE_LOCALLY:
            download.download_papyrus(
                outdir=self.papyrus_params['source_path'],
                # pystow_path_key (not .version): must match the folder key
                # every read in this class looks up self.papyrus_params['version']
                # under, or an old-format pv silently downloads to the wrong folder.
                version=self.papyrus_params['version'].pystow_path_key,
                nostereo=not self.papyrus_params['is3d'],
                stereo=self.papyrus_params['is3d'],
                only_pp=self.papyrus_params['plusplus'],
                structures=False,
                descriptors=desc_type,
                progress=self.papyrus_params['download_progress'],
                disk_margin=self.papyrus_params['disk_margin'],
                keep_xz=self.papyrus_params['keep_original_files'],
            )
            return self.protein_descriptors(desc_type, progress)


# ---------------------------------------------------------------------------
# PapyrusProteinSet
# ---------------------------------------------------------------------------

class PapyrusProteinSet(ProteinSet):
    """A set of protein targets derived from a :class:`PapyrusDataset`."""

    def __init__(
            self,
            df: pl.DataFrame | Iterator,
            papyrus_params: dict,
            num_proteins: int,
    ) -> None:
        """Wrap already-loaded protein data (or a chunk iterator) plus its row count."""
        self.data = df
        self.papyrus_params = papyrus_params
        self.num_rows = num_proteins

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def aggregate(self, progress: bool = False) -> pl.DataFrame:
        """Materialise the protein targets into a :class:`~polars.DataFrame`.

        :param progress: show a progress bar
        """
        if isinstance(self.data, pl.DataFrame):
            return self.data
        total = _num_chunks(self.num_rows, self.papyrus_params['chunksize'])
        return preprocess.consume_chunks(generator=self.data, progress=progress, total=total)

    def agg(self, progress: bool = False) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def consume_chunks(self, progress: bool = False) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def to_dataframe(self, progress: bool = False) -> pl.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Show the protein count, or that materialisation is still pending."""
        if not isinstance(self.data, pl.DataFrame):
            return f'{type(self).__name__}<iterator of proteins>'
        return f'{type(self).__name__}<{len(self.data)} proteins>'


# ---------------------------------------------------------------------------
# PapyrusPDBProteinSet
# ---------------------------------------------------------------------------

class PapyrusPDBProteinSet(ProteinSet):
    """A set of RCSB PDB protein structures matched to a :class:`PapyrusDataset`."""

    def __init__(
            self,
            df: pd.DataFrame | Iterator,
            papyrus_params: dict,
    ) -> None:
        """Wrap already-loaded PDB structure data (or a chunk iterator)."""
        self.data = df
        self.papyrus_params = papyrus_params
        self.num_rows: int | None = len(df) if isinstance(df, pd.DataFrame) else None

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        """Materialise the PDB structures into a :class:`~pandas.DataFrame`.

        :param progress: show a progress bar
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data
        total = _num_chunks(self.num_rows, self.papyrus_params.get('chunksize'))
        return preprocess.consume_chunks(generator=self.data, progress=progress, total=total)

    def agg(self, progress: bool = False) -> pd.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def consume_chunks(self, progress: bool = False) -> pd.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    def to_dataframe(self, progress: bool = False) -> pd.DataFrame:
        """Alias for :meth:`aggregate`."""
        return self.aggregate(progress=progress)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Show the structure count, or that materialisation is still pending."""
        if not isinstance(self.data, pd.DataFrame):
            return f'{type(self).__name__}<iterator of protein structures>'
        return f'{type(self).__name__}<{len(self.data)} protein structures>'
