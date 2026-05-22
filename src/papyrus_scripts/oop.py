# -*- coding: utf-8 -*-

"""Object-oriented API for the Papyrus dataset."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import prodec
import pystow

from . import download, reader, preprocess, subsim_search
from .fingerprint import Fingerprint, MorganFingerprint
from .matchRCSB import get_matches as get_pdb_matches
from .utils import IO


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_papyrus_version(version: Union[str, IO.PapyrusVersion]) -> IO.PapyrusVersion:
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


def _num_chunks(num_rows: int, chunksize: Optional[int]) -> Optional[int]:
    """Return the number of chunks or ``None`` when *chunksize* is ``None``."""
    if chunksize is None or num_rows is None:
        return None
    return _ceil_div(num_rows, chunksize)


def _id_column(is3d: bool) -> str:
    """Return the molecule-identifier column name for the given stereo flag."""
    return 'InChIKey' if is3d else 'connectivity'


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

    def __init__(
            self,
            version: Union[str, IO.PapyrusVersion] = 'latest',
            is3d: bool = False,
            plusplus: bool = True,
            chunksize: Optional[int] = 1_000_000,
            source_path: Optional[str] = None,
            download_progress: bool = False,
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
        :raises ValueError: if both *is3d* and *plusplus* are True (the 3D
            Papyrus++ combination does not exist)
        """
        pv = _ensure_papyrus_version(version)

        # Auto-download if the data is not available locally.
        if not IO.is_local_version_available(version=pv, root_folder=source_path):
            download.download_papyrus(
                outdir=source_path,
                version=pv.version_old_fmt,
                nostereo=True, stereo=True, only_pp=False,
                structures=True, descriptors='all',
                progress=download_progress, disk_margin=0.0,
            )

        self.papyrus_params: Dict = dict(
            is3d=is3d,
            version=pv,
            plusplus=plusplus,
            chunksize=chunksize,
            source_path=source_path,
            num_rows=IO.get_num_rows_in_file(
                filetype='bioactivities', is3D=is3d,
                version=pv, plusplus=plusplus,
                root_folder=source_path,
            ),
            download_progress=download_progress,
        )
        self.papyrus_bioactivity_data = reader.read_papyrus(
            is3d=is3d, version=pv, plusplus=plusplus,
            chunksize=chunksize, source_path=source_path,
        )
        self.papyrus_protein_data = reader.read_protein_set(
            source_path=source_path, version=pv,
        )
        self._fpsubsim2_: Optional[FPSubSim2Engine] = None
        self._can_reset: bool = True

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def from_dataframe(
            df: pd.DataFrame,
            is3d: bool,
            version: Union[str, IO.PapyrusVersion],
            plusplus: bool = True,
            source_path: Optional[str] = None,
            download_progress: bool = False,
            chunksize: Optional[int] = None,
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
        :returns: a :class:`PapyrusDataset` wrapping *df*
        """
        pv = _ensure_papyrus_version(version)
        dataset = PapyrusDataset.__new__(PapyrusDataset)
        dataset.papyrus_bioactivity_data = df
        dataset.papyrus_protein_data = reader.read_protein_set(
            source_path=source_path, version=pv,
        )
        dataset.papyrus_params = dict(
            is3d=is3d, version=pv, plusplus=plusplus,
            chunksize=chunksize, source_path=source_path,
            num_rows=len(df), download_progress=download_progress,
        )
        dataset._fpsubsim2_: Optional[FPSubSim2Engine] = None
        dataset._can_reset: bool = False
        return dataset

    @staticmethod
    def _from_data(
            papyrus_bioactivity_data: Union[Iterator[pd.DataFrame], pd.DataFrame],
            papyrus_protein_data: pd.DataFrame,
            papyrus_params: Dict,
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
        dataset._fpsubsim2_: Optional[FPSubSim2Engine] = None
        dataset._can_reset: bool = False
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

    def keep_source(self, source: Union[List[str], str]) -> PapyrusDataset:
        """Keep samples from specific data source(s).

        :param source: source label(s) such as ``'chembl'`` or
            ``['chembl', 'klaeger']``
        """
        return self._filter.keep_source(source=source)

    def keep_activity_type(self, activity_types: Union[List[str], str]) -> PapyrusDataset:
        """Keep samples of specific activity type(s).

        :param activity_types: type(s) such as ``'ic50'`` or
            ``['ki', 'ec50']``
        """
        return self._filter.keep_activity_type(activity_types=activity_types)

    def keep_accession(self, accession: Union[List[str], str] = 'all') -> PapyrusDataset:
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
            classes: Optional[Union[dict, List[dict]]],
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
            organism: Optional[Union[str, List[str]]],
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

    def isin(self, column: str, values: Union[Any, List[Any]]) -> PapyrusDataset:
        """Keep samples whose *column* value is in *values*.

        :param column: column to inspect
        :param values: acceptable value(s)
        """
        return self._filter.isin(column=column, values=values)

    def not_isin(self, column: str, values: Union[Any, List[Any]]) -> PapyrusDataset:
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
            smiles: Union[str, List[str]],
            fp: Fingerprint = None,
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
                                                      threshold=threshold, cuda=cuda
                                                      )

    def keep_dissimilar_molecules(
            self,
            smiles: Union[str, List[str]],
            fp: Fingerprint = None,
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
                                                         threshold=threshold, cuda=cuda
                                                         )

    def keep_substructure_molecules(self, smiles: Union[str, List[str]]) -> PapyrusDataset:
        """Keep samples whose structures contain any of the given SMILES as a substructure.

        :param smiles: query SMILES string(s)
        """
        return self._fpsubsim2.keep_substructure_molecules(smiles=smiles)

    def keep_not_substructure_molecules(self, smiles: Union[str, List[str]]) -> PapyrusDataset:
        """Keep samples whose structures do **not** contain any of the given SMILES as a substructure.

        :param smiles: query SMILES string(s)
        """
        return self._fpsubsim2.keep_not_substructure_molecules(smiles=smiles)

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        """Materialise all lazy filters into a single :class:`~pandas.DataFrame`.

        :param progress: show a tqdm progress bar while consuming chunks
        :returns: a DataFrame of the filtered data
        """
        if isinstance(self.papyrus_bioactivity_data, pd.DataFrame):
            return self.papyrus_bioactivity_data
        total = _num_chunks(self.papyrus_params['num_rows'], self.papyrus_params['chunksize'])
        return preprocess.consume_chunks(
            generator=self.papyrus_bioactivity_data, progress=progress, total=total,
        )

    #: Alias for :meth:`aggregate`.
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
    # Derived datasets
    # ------------------------------------------------------------------

    def molecules(
            self,
            chunksize: Optional[int] = 1_000_000,
            progress: bool = False,
    ) -> PapyrusMoleculeSet:
        """Return the molecular structures for the samples in this dataset.

        :param chunksize: structures per chunk (default: 1 000 000)
        :param progress: show progress while aggregating the bioactivity data
        :returns: a :class:`PapyrusMoleculeSet`
        """
        ids = self.aggregate(progress=progress)[_id_column(self.papyrus_params['is3d'])].unique()
        molecules = reader.read_molecular_structures(
            is3d=self.papyrus_params['is3d'],
            version=self.papyrus_params['version'],
            chunksize=chunksize,
            source_path=self.papyrus_params['source_path'],
            ids=ids, verbose=False,
        )
        return PapyrusMoleculeSet(molecules, {**self.papyrus_params, 'chunksize': chunksize})

    def proteins(self, progress: bool = False) -> PapyrusProteinSet:
        """Return the protein targets for the samples in this dataset.

        :param progress: show progress while aggregating the bioactivity data
        :returns: a :class:`PapyrusProteinSet`
        """
        ids = self.aggregate(progress=progress)['target_id'].unique()
        proteins = self.papyrus_protein_data[
            self.papyrus_protein_data['target_id'].isin(ids)
        ]
        return PapyrusProteinSet(proteins, self.papyrus_params, num_proteins=len(proteins))

    def match_rcsb_pdb(self, update: bool = True, progress: bool = False) -> PapyrusPDBProteinSet:
        """Match samples to RCSB Protein Data Bank 3D structures.

        :param update: refresh the local PDB identifier cache (default: True)
        :param progress: show progress while matching
        :returns: a :class:`PapyrusPDBProteinSet`
        """
        total = _num_chunks(self.papyrus_params['num_rows'], self.papyrus_params['chunksize'])
        structures = get_pdb_matches(
            self.papyrus_bioactivity_data,
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
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Return molecular descriptors for the molecules in this dataset.

        Downloads the descriptor file if it is not yet available locally.

        :param desc_type: descriptor set; one of ``'mold2'``, ``'mordred'``,
            ``'cddd'``, ``'fingerprint'``, ``'moe'``, ``'all'``
        :param progress: show progress while aggregating
        :returns: a DataFrame (or lazy iterator) of molecular descriptors
        """
        ids = self.aggregate(progress)[_id_column(self.papyrus_params['is3d'])].unique()
        try:
            return reader.read_molecular_descriptors(
                desc_type=desc_type,
                is3d=self.papyrus_params['is3d'],
                version=self.papyrus_params['version'],
                chunksize=self.papyrus_params['chunksize'],
                source_path=self.papyrus_params['source_path'],
                ids=ids,
                verbose=progress,
            )
        except FileNotFoundError:
            download.download_papyrus(
                outdir=self.papyrus_params['source_path'],
                version=self.papyrus_params['version'].version_old_fmt,
                nostereo=not self.papyrus_params['is3d'],
                stereo=self.papyrus_params['is3d'],
                only_pp=self.papyrus_params['plusplus'],
                structures=False,
                descriptors=desc_type,
                progress=self.papyrus_params['download_progress'],
                disk_margin=0.0,
            )
            return self.molecular_descriptors(desc_type, progress)

    # ------------------------------------------------------------------
    # Administration
    # ------------------------------------------------------------------

    def reset(self) -> bool:
        """Reset the underlying data stream to the beginning of the file.

        Has no effect when the dataset was created from a DataFrame.

        :returns: ``True`` if the stream was reset, ``False`` otherwise
        """
        if self._can_reset:
            self.papyrus_bioactivity_data = reader.read_papyrus(
                is3d=self.papyrus_params['is3d'],
                version=self.papyrus_params['version'],
                plusplus=self.papyrus_params['plusplus'],
                chunksize=self.papyrus_params['chunksize'],
                source_path=self.papyrus_params['source_path'],
            )
            self.papyrus_protein_data = reader.read_protein_set(
                source_path=self.papyrus_params['source_path'],
                version=self.papyrus_params['version'],
            )
        return self._can_reset

    @staticmethod
    def remove(
            version: Union[str, IO.PapyrusVersion],
            remove_papyruspp: bool,
            remove_bioactivities: bool,
            remove_proteins: bool,
            remove_nostereo: bool,
            remove_stereo: bool,
            remove_structures: bool,
            remove_descriptors: Union[str, List[str]],
            remove_other_files: bool,
            remove_version_root: bool,
            remove_papyrus_root: bool,
            force: bool = False,
            progress: bool = True,
            source_path: Optional[str] = None,
    ) -> None:
        """Remove locally downloaded Papyrus data.

        All arguments map directly to :func:`~download.remove_papyrus`; see
        that function for full documentation.
        """
        pv = _ensure_papyrus_version(version)
        download.remove_papyrus(
            outdir=source_path,
            version=pv.version_old_fmt,
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
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params = ', '.join(f'{k}={v}' for k, v in self.papyrus_params.items())
        return f'{type(self).__name__}<{params}>'


# ---------------------------------------------------------------------------
# PapyrusDataFilter
# ---------------------------------------------------------------------------

class PapyrusDataFilter:
    """Collection of filters applied to a :class:`PapyrusDataset`.

    Normally obtained via :attr:`PapyrusDataset._filter`.  Can be configured
    for parallelism and verbosity via :meth:`__call__` before chaining a
    filter method::

        dataset._filter(njobs=4, progress=True).keep_quality('medium')
    """

    def __init__(
            self,
            papyrus_bioactivity_data: Union[Iterator[pd.DataFrame], pd.DataFrame],
            papyrus_protein_data: pd.DataFrame,
            papyrus_params: Dict,
            njobs: int = 1,
            progress: bool = False,
    ) -> None:
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

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def keep_quality(self, min_quality: str = 'high') -> PapyrusDataset:
        return self._wrap(preprocess.keep_quality(
            data=self.papyrus_bioactivity_data, min_quality=min_quality,
        )
        )

    def keep_source(self, source: Union[List[str], str] = 'all') -> PapyrusDataset:
        return self._wrap(preprocess.keep_source(
            data=self.papyrus_bioactivity_data, source=source,
            njobs=self.njobs, verbose=self.progress,
        )
        )

    def keep_activity_type(self, activity_types: Union[List[str], str] = 'ic50') -> PapyrusDataset:
        return self._wrap(preprocess.keep_type(
            data=self.papyrus_bioactivity_data, activity_types=activity_types,
            njobs=self.njobs, verbose=self.progress,
        )
        )

    def keep_accession(self, accession: Union[List[str], str] = 'all') -> PapyrusDataset:
        return self._wrap(preprocess.keep_accession(
            data=self.papyrus_bioactivity_data, accession=accession,
        )
        )

    def keep_protein_class(
            self,
            classes: Optional[Union[dict, List[dict]]],
            generic_regex: bool = False,
    ) -> PapyrusDataset:
        return self._wrap(preprocess.keep_protein_class(
            data=self.papyrus_bioactivity_data,
            protein_data=self.papyrus_protein_data,
            classes=classes, generic_regex=generic_regex,
        )
        )

    def keep_organism(
            self,
            organism: Optional[Union[str, List[str]]] = 'Homo sapiens (Human)',
            generic_regex: bool = False,
    ) -> PapyrusDataset:
        return self._wrap(preprocess.keep_organism(
            data=self.papyrus_bioactivity_data,
            protein_data=self.papyrus_protein_data,
            organism=organism, generic_regex=generic_regex,
        )
        )

    def contains(
            self, column: str, value: str, case: bool = True, regex: bool = False,
    ) -> PapyrusDataset:
        return self._wrap(preprocess.keep_contains(
            data=self.papyrus_bioactivity_data,
            column=column, value=value, case=case, regex=regex,
        )
        )

    def not_contains(
            self, column: str, value: str, case: bool = True, regex: bool = False,
    ) -> PapyrusDataset:
        return self._wrap(preprocess.keep_not_contains(
            data=self.papyrus_bioactivity_data,
            column=column, value=value, case=case, regex=regex,
        )
        )

    def isin(self, column: str, values: Union[Any, List[Any]]) -> PapyrusDataset:
        return self._wrap(preprocess.keep_match(
            data=self.papyrus_bioactivity_data, column=column, values=values,
        )
        )

    def not_isin(self, column: str, values: Union[Any, List[Any]]) -> PapyrusDataset:
        return self._wrap(preprocess.keep_not_match(
            data=self.papyrus_bioactivity_data, column=column, values=values,
        )
        )


# ---------------------------------------------------------------------------
# FPSubSim2Engine
# ---------------------------------------------------------------------------

class FPSubSim2Engine:
    """Manages creation, loading, and querying of an FPSubSim2 similarity /
    substructure search database for a specific :class:`PapyrusDataset`.
    """

    def __init__(self, papyrus_params: Dict) -> None:
        self.papyrus_params = papyrus_params
        self.path: Optional[str] = None
        self.progress: bool = False
        self.fp: Fingerprint = MorganFingerprint()
        self.fpsubsim2 = subsim_search.FPSubSim2()
        self.papyrus_bioactivity_data = None
        self.papyrus_protein_data = None

    def __call__(
            self,
            fp: Optional[Union[Fingerprint, List[Fingerprint]]] = None,
            path: Optional[str] = None,
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

    def _resolve_path(self) -> str:
        """Derive the default ``.h5`` path from ``papyrus_params``.

        :returns: absolute path to the FPSubSim2 database file
        :raises NotADirectoryError: if the parent directory does not exist
        """
        pv = self.papyrus_params['version']  # PapyrusVersion
        is3d = self.papyrus_params['is3d']
        dim_tag = '3D' if is3d else '2D'
        stereo = 'without' if not is3d else 'with'
        name = (f'{pv.version_old_fmt}_combined_set_'
                f'{stereo}_stereochemistry_FPSubSim2_{dim_tag}.h5')

        if self.papyrus_params['source_path'] is not None:
            os.environ['PYSTOW_HOME'] = os.path.abspath(self.papyrus_params['source_path'])

        path = pystow.module('papyrus', pv.version_old_fmt).join(name=name).as_posix()

        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            raise NotADirectoryError(
                f'Cannot create the FPSubSim2 file in a non-existing folder: {parent!r}'
            )
        return path

    def _ensure_loaded(self) -> None:
        """Load or create the FPSubSim2 database as needed."""
        if self.path is None:
            self.path = self._resolve_path()

        if os.path.isfile(self.path):
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
            papyrus_bioactivity_data: Union[Iterator[pd.DataFrame], pd.DataFrame],
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

    # ------------------------------------------------------------------
    # Public filter methods
    # ------------------------------------------------------------------

    def keep_similar_molecules(
            self,
            smiles: Union[str, List[str]],
            fp: Optional[Fingerprint] = None,
            threshold: float = 0.7,
            cuda: bool = False,
    ) -> PapyrusDataset:
        """Keep samples similar to any of the query SMILES."""
        self._ensure_loaded()
        return self._wrap(preprocess.keep_similar(
            data=self.papyrus_bioactivity_data,
            molecule_smiles=smiles,
            fpsubsim2_file=self.path,
            fingerprint=fp if fp is not None else MorganFingerprint(),
            threshold=threshold,
            cuda=cuda,
        )
        )

    def keep_dissimilar_molecules(
            self,
            smiles: Union[str, List[str]],
            fp: Optional[Fingerprint] = None,
            threshold: float = 0.7,
            cuda: bool = False,
    ) -> PapyrusDataset:
        """Keep samples **not** similar to any of the query SMILES."""
        self._ensure_loaded()
        return self._wrap(preprocess.keep_dissimilar(
            data=self.papyrus_bioactivity_data,
            molecule_smiles=smiles,
            fpsubsim2_file=self.path,
            fingerprint=fp if fp is not None else MorganFingerprint(),
            threshold=threshold,
            cuda=cuda,
        )
        )

    def keep_substructure_molecules(self, smiles: Union[str, List[str]]) -> PapyrusDataset:
        """Keep samples that are substructures of any of the query SMILES."""
        self._ensure_loaded()
        return self._wrap(preprocess.keep_substructure(
            data=self.papyrus_bioactivity_data,
            molecule_smiles=smiles,
            fpsubsim2_file=self.path,
        )
        )

    def keep_not_substructure_molecules(self, smiles: Union[str, List[str]]) -> PapyrusDataset:
        """Keep samples that are **not** substructures of any of the query SMILES."""
        self._ensure_loaded()
        return self._wrap(preprocess.keep_not_substructure(
            data=self.papyrus_bioactivity_data,
            molecule_smiles=smiles,
            fpsubsim2_file=self.path,
        )
        )


# ---------------------------------------------------------------------------
# PapyrusMoleculeSet
# ---------------------------------------------------------------------------

class PapyrusMoleculeSet:
    """A set of molecular structures derived from a :class:`PapyrusDataset`."""

    def __init__(
            self,
            df: Union[pd.DataFrame, Iterator],
            papyrus_params: Dict,
    ) -> None:
        self.data = df
        self.papyrus_params = papyrus_params
        self.num_rows = IO.get_num_rows_in_file(
            filetype='structures',
            is3D=self.papyrus_params['is3d'],
            version=self.papyrus_params['version'],
            root_folder=self.papyrus_params['source_path'],
        )

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        """Materialise all structures into a single :class:`~pandas.DataFrame`.

        :param progress: show a progress bar
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data
        total = _num_chunks(self.num_rows, self.papyrus_params['chunksize'])
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
    # Descriptors
    # ------------------------------------------------------------------

    def molecular_descriptors(
            self,
            desc_type: str,
            progress: bool = False,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Return molecular descriptors for the molecules in this set.

        Downloads the descriptor file if not yet available locally.

        :param desc_type: one of ``'mold2'``, ``'mordred'``, ``'cddd'``,
            ``'fingerprint'``, ``'moe'``, ``'all'``
        :param progress: show progress while aggregating
        """
        ids = self.aggregate(progress)[_id_column(self.papyrus_params['is3d'])].unique()
        try:
            return reader.read_molecular_descriptors(
                desc_type=desc_type,
                is3d=self.papyrus_params['is3d'],
                version=self.papyrus_params['version'],
                chunksize=self.papyrus_params['chunksize'],
                source_path=self.papyrus_params['source_path'],
                ids=ids,
                verbose=progress,
            )
        except FileNotFoundError:
            download.download_papyrus(
                outdir=self.papyrus_params['source_path'],
                version=self.papyrus_params['version'].version_old_fmt,
                nostereo=not self.papyrus_params['is3d'],
                stereo=self.papyrus_params['is3d'],
                only_pp=self.papyrus_params['plusplus'],
                structures=False,
                descriptors=desc_type,
                progress=self.papyrus_params['download_progress'],
                disk_margin=0.0,
            )
            return self.molecular_descriptors(desc_type, progress)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not isinstance(self.data, pd.DataFrame):
            return f'{type(self).__name__}<iterator of molecules>'
        return f'{type(self).__name__}<{len(self.data)} molecules>'


# ---------------------------------------------------------------------------
# ProteinSet  (abstract base)
# ---------------------------------------------------------------------------

class ProteinSet(ABC):
    """Abstract base for protein-target set classes."""

    data: pd.DataFrame
    papyrus_params: Dict

    @abstractmethod
    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        """Materialise the protein data into a DataFrame."""

    def protein_descriptors(
            self,
            desc_type: Union[str, prodec.Descriptor, prodec.Transform],
            progress: bool = False,
    ) -> pd.DataFrame:
        """Return protein descriptors for the targets in this set.

        Downloads the descriptor file if not yet available locally.

        :param desc_type: descriptor type: ``'unirep'``, ``'custom'``, or a
            ProDEC :class:`~prodec.Descriptor` / :class:`~prodec.Transform`
        :param progress: show progress while aggregating
        """
        self.data = self.aggregate(progress)
        ids = self.data['target_id'].unique()
        try:
            return reader.read_protein_descriptors(
                desc_type=desc_type,
                version=self.papyrus_params['version'],
                chunksize=self.papyrus_params['chunksize'],
                source_path=self.papyrus_params['source_path'],
                ids=ids,
                verbose=progress,
            )
        except FileNotFoundError:
            download.download_papyrus(
                outdir=self.papyrus_params['source_path'],
                version=self.papyrus_params['version'].version_old_fmt,
                nostereo=not self.papyrus_params['is3d'],
                stereo=self.papyrus_params['is3d'],
                only_pp=self.papyrus_params['plusplus'],
                structures=False,
                descriptors=desc_type,
                progress=self.papyrus_params['download_progress'],
                disk_margin=0.0,
            )
            return self.protein_descriptors(desc_type, progress)


# ---------------------------------------------------------------------------
# PapyrusProteinSet
# ---------------------------------------------------------------------------

class PapyrusProteinSet(ProteinSet):
    """A set of protein targets derived from a :class:`PapyrusDataset`."""

    def __init__(
            self,
            df: Union[pd.DataFrame, Iterator],
            papyrus_params: Dict,
            num_proteins: int,
    ) -> None:
        self.data = df
        self.papyrus_params = papyrus_params
        self.num_rows = num_proteins

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        """Materialise the protein targets into a :class:`~pandas.DataFrame`.

        :param progress: show a progress bar
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data
        total = _num_chunks(self.num_rows, self.papyrus_params['chunksize'])
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
        if not isinstance(self.data, pd.DataFrame):
            return f'{type(self).__name__}<iterator of proteins>'
        return f'{type(self).__name__}<{len(self.data)} proteins>'


# ---------------------------------------------------------------------------
# PapyrusPDBProteinSet
# ---------------------------------------------------------------------------

class PapyrusPDBProteinSet(ProteinSet):
    """A set of RCSB PDB protein structures matched to a :class:`PapyrusDataset`."""

    def __init__(
            self,
            df: Union[pd.DataFrame, Iterator],
            papyrus_params: Dict,
    ) -> None:
        self.data = df
        self.papyrus_params = papyrus_params
        self.num_rows: Optional[int] = len(df) if isinstance(df, pd.DataFrame) else None

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
        if not isinstance(self.data, pd.DataFrame):
            return f'{type(self).__name__}<iterator of protein structures>'
        return f'{type(self).__name__}<{len(self.data)} protein structures>'
