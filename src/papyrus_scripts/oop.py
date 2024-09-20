# -*- coding: utf-8 -*-

"""Reading capacities of the Papyrus-scripts."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Union, Optional

import pystow
import pandas as pd
import prodec

from . import download
from . import reader
from . import fingerprint
from . import preprocess
from . import subsim_search
from .utils import IO
from .matchRCSB import get_matches as get_pdb_matches


class PapyrusDataset:
    """Papyrus dataset to facilitate data access and filtering."""

    def __init__(self, version: str | IO.PapyrusVersion = 'latest', is3d: bool = False, plusplus: bool = True,
                 chunksize: Optional[int] = 1_000_000, source_path: Optional[str] = None,
                 download_progress: bool = False):
        version = IO.PapyrusVersion(version=version)
        if not IO.is_local_version_available(version=version.version_old_fmt, root_folder=source_path):
            download.download_papyrus(outdir=source_path, version=version.version_old_fmt, nostereo=True, stereo=True,
                                      only_pp=False, structures=True, descriptors='all', progress=download_progress,
                                      disk_margin=0.0)
        self.papyrus_params = dict(is3d=is3d, version=version, plusplus=plusplus,
                                   chunksize=chunksize, source_path=source_path,
                                   num_rows=IO.get_num_rows_in_file(filetype='bioactivities', is3D=is3d,
                                                                    version=version, plusplus=plusplus,
                                                                    root_folder=source_path),
                                   download_progress=download_progress)
        self.papyrus_bioactivity_data = reader.read_papyrus(is3d=is3d, version=version, plusplus=plusplus,
                                                            chunksize=chunksize, source_path=source_path)
        self.papyrus_protein_data = reader.read_protein_set(source_path=source_path, version=version)
        self._fpsubsim2_ = None
        self._can_reset = True

    @staticmethod
    def from_dataframe(df: pd.DataFrame,
                       is3d: bool, version: str,
                       plusplus: bool = True,
                       source_path: Optional[str] = None,
                       download_progress: bool = False
                       ) -> PapyrusDataset:
        dataset = PapyrusDataset.__new__(PapyrusDataset)
        dataset.papyrus_bioactivity_data = df
        dataset.papyrus_protein_data = reader.read_protein_set(source_path=source_path, version=version)
        dataset.papyrus_params = dict(is3d=is3d, version=version, plusplus=plusplus,
                                      chunksize=len(df), source_path=source_path, num_rows=len(df),
                                      download_progress=download_progress)
        dataset._can_reset = False
        return dataset

    @staticmethod
    def _from_data(papyrus_bioactivity_data: Union[Iterator[pd.DataFrame], pd.DataFrame],
                   papyrus_protein_data: pd.DataFrame,
                   papyrus_params: Dict
                   ) -> PapyrusDataset:
        dataset = PapyrusDataset.__new__(PapyrusDataset)
        dataset.papyrus_bioactivity_data = papyrus_bioactivity_data
        dataset.papyrus_protein_data = papyrus_protein_data
        dataset.papyrus_params = papyrus_params
        dataset._can_reset = False
        return dataset

    @property
    def _filter(self) -> PapyrusDataFilter:
        return PapyrusDataFilter(papyrus_bioactivity_data=self.papyrus_bioactivity_data,
                                 papyrus_protein_data=self.papyrus_protein_data,
                                 papyrus_params=self.papyrus_params)

    @property
    def _fpsubsim2(self) -> FPSubSim2Engine:
        if self._fpsubsim2_ is None:
            self._fpsubsim2_ = FPSubSim2Engine(self.papyrus_params)
        self._fpsubsim2_._set_data(papyrus_bioactivity_data=self.papyrus_bioactivity_data,
                                  papyrus_protein_data=self.papyrus_protein_data)
        return self._fpsubsim2_

    def keep_quality(self, min_quality: str) -> PapyrusDataset:
        return self._filter.keep_quality(min_quality=min_quality)

    def keep_source(self, source: Union[List[str], str]) -> PapyrusDataset:
        return self._filter.keep_source(source=source)

    def keep_activity_type(self, activity_types: Union[List[str], str]) -> PapyrusDataset:
        return self._filter.keep_activity_type(activity_types=activity_types)

    def keep_accession(self, accession: Union[List[str], str] = 'all') -> PapyrusDataset:
        return self._filter.keep_accession(accession=accession)

    def keep_protein_class(self, classes: Optional[Union[dict, List[dict]]],
                           generic_regex: bool = False) -> PapyrusDataset:
        return self._filter.keep_protein_class(classes=classes, generic_regex=generic_regex)

    def keep_organism(self, organism: Optional[Union[str, List[str]]],
                      generic_regex: bool = False) -> PapyrusDataset:
        return self._filter.keep_organism(organism=organism, generic_regex=generic_regex)

    def contains(self, column: str, value: str, case: bool = True, regex: bool = False) -> PapyrusDataset:
        return self._filter.contains(column=column, value=value, case=case, regex=regex)

    def not_contains(self, column: str, value: str, case: bool = True, regex: bool = False) -> PapyrusDataset:
        return self._filter.not_contains(column=column, value=value, case=case, regex=regex)

    def isin(self, column: str, values: Union[Any, List[Any]]) -> PapyrusDataset:
        return self._filter.isin(column=column, values=values)

    def not_isin(self, column: str, values: Union[Any, List[Any]]) -> PapyrusDataset:
        return self._filter.not_isin(column=column, values=values)

    def keep_similar_molecules(self, smiles: Union[str, List[str]],
                               fingerprint: fingerprint.Fingerprint = fingerprint.MorganFingerprint(),
                               threshold: float = 0.7, cuda: bool = False) -> PapyrusDataset:
        return self._fpsubsim2.keep_similar_molecules(smiles=smiles, fingerprint=fingerprint, threshold=threshold,
                                                      cuda=cuda)

    def keep_dissimilar_molecules(self, smiles: Union[str, List[str]],
                                  fingerprint: fingerprint.Fingerprint = fingerprint.MorganFingerprint(),
                                  threshold: float = 0.7, cuda: bool = False) -> PapyrusDataset:
        return self._fpsubsim2.keep_dissimilar_molecules(smiles=smiles, fingerprint=fingerprint, threshold=threshold,
                                                         cuda=cuda)

    def keep_substructure_molecules(self, smiles: Union[str, List[str]]) -> PapyrusDataset:
        return self._fpsubsim2.keep_substructure_molecules(smiles=smiles)

    def keep_not_substructure_molecules(self, smiles: Union[str, List[str]]) -> PapyrusDataset:
        return self._fpsubsim2.keep_not_substructure_molecules(smiles=smiles)

    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        total = (-(-self.papyrus_params['num_rows'] // self.papyrus_params['chunksize'])
                 if self.papyrus_params['chunksize'] is not None
                 else None)
        if isinstance(self.papyrus_bioactivity_data, pd.DataFrame):
            return self.papyrus_bioactivity_data
        return preprocess.consume_chunks(generator=self.papyrus_bioactivity_data,
                                         progress=progress, total=total)

    def agg(self, progress: bool = False) -> pd.DataFrame:
        return self.aggregate(progress=progress)

    def consume_chunks(self, progress: bool = False) -> pd.DataFrame:
        return self.aggregate(progress=progress)

    def to_dataframe(self, progress: bool = False) -> pd.DataFrame:
        return self.aggregate(progress=progress)

    def molecules(self, chunksize: Optional[int] = 1_000_000, progress: bool = False) -> PapyrusMoleculeSet:
        ids = self.aggregate(progress=progress)['connectivity' if not self.papyrus_params['is3d'] else 'InChIKey'].unique()
        molecules = reader.read_molecular_structures(is3d=self.papyrus_params['is3d'],
                                                     version=self.papyrus_params['version'],
                                                     chunksize=chunksize,
                                                     source_path=self.papyrus_params['source_path'],
                                                     ids=ids, verbose=False)
        return PapyrusMoleculeSet(molecules, {**self.papyrus_params, 'chunksize': chunksize})

    def proteins(self, progress: bool = False) -> PapyrusProteinSet:
        ids = self.aggregate(progress=progress)['target_id'].unique()
        proteins = self.papyrus_protein_data[self.papyrus_protein_data.target_id.isin(ids)]
        return PapyrusProteinSet(proteins, self.papyrus_params,
                                 len(proteins))

    def match_rcsb_pdb(self, update: bool = True, progress: bool = False) -> PapyrusPDBProteinSet:
        total = (-(-self.papyrus_params['num_rows'] // self.papyrus_params['chunksize'])
                 if self.papyrus_params['chunksize'] is not None
                 else None)
        structures = get_pdb_matches(self.papyrus_bioactivity_data, root_folder=self.papyrus_params['source_path'],
                                     verbose=progress, total=total, update=update)
        return PapyrusPDBProteinSet(structures)

    def __repr__(self):
        return f'{type(self).__name__}<{", ".join(f"{key}={value}" for key, value in self.papyrus_params.items())}>'

    def reset(self):
        """Reset the underlying data stream if not instantiated from a dataframe and return if was reset."""
        if self._can_reset:
            self.papyrus_bioactivity_data = reader.read_papyrus(is3d=self.papyrus_params['is3d'],
                                                                version=self.papyrus_params['version'],
                                                                plusplus=self.papyrus_params['plusplus'],
                                                                chunksize=self.papyrus_params['chunksize'],
                                                                source_path=self.papyrus_params['source_path'])
            self.papyrus_protein_data = reader.read_protein_set(source_path=self.papyrus_params['source_path'],
                                                                version=self.papyrus_params['version'])
        return self._can_reset

    @staticmethod
    def remove(version: str,
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
               source_path: Optional[str] = None) -> None:
        download.remove_papyrus(outdir=source_path, version=version, papyruspp=remove_papyruspp,
                                bioactivities=remove_bioactivities, proteins=remove_proteins,
                                nostereo=remove_nostereo, stereo=remove_stereo, structures=remove_structures,
                                descriptors=remove_descriptors, other_files=remove_other_files,
                                version_root=remove_version_root, papyrus_root=remove_papyrus_root,
                                force=force, progress=progress)

class PapyrusDataFilter:
    """Collection of filters to be applied on a PapyrusDataset instance."""

    def __init__(self,
                 papyrus_bioactivity_data: Union[Iterator[pd.DataFrame], pd.DataFrame],
                 papyrus_protein_data: pd.DataFrame,
                 papyrus_params: Dict,
                 njobs: int = 1,
                 progress: bool = False):
        self.papyrus_bioactivity_data = papyrus_bioactivity_data
        self.papyrus_protein_data = papyrus_protein_data
        self.papyrus_params = papyrus_params
        self.njobs = njobs
        self.progress = progress

    def __call__(self, njobs: int = 1, progress: bool = False):
        self.njobs = njobs
        self.progress = progress

    def keep_quality(self, min_quality: str = 'high') -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_quality(data=self.papyrus_bioactivity_data,
                                                             min_quality=min_quality),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def keep_source(self, source: Union[List[str], str] = 'all') -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_source(data=self.papyrus_bioactivity_data, source=source,
                                                            njobs=self.njobs, verbose=self.progress),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def keep_activity_type(self, activity_types: Union[List[str], str] = 'ic50') -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_type(data=self.papyrus_bioactivity_data,
                                                          activity_types=activity_types, njobs=self.njobs,
                                                          verbose=self.progress),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def keep_accession(self, accession: Union[List[str], str] = 'all') -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_accession(data=self.papyrus_bioactivity_data, accession=accession),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def keep_protein_class(self,
                           classes: Optional[Union[dict, List[dict]]],
                           generic_regex: bool = False
                           ) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_protein_class(data=self.papyrus_bioactivity_data,
                                                                   protein_data=self.papyrus_protein_data,
                                                                   classes=classes, generic_regex=generic_regex),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def keep_organism(self, organism: Optional[Union[str, List[str]]] = 'Homo sapiens (Human)',
                      generic_regex: bool = False) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_organism(data=self.papyrus_bioactivity_data,
                                                              protein_data=self.papyrus_protein_data, organism=organism,
                                                              generic_regex=generic_regex),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def contains(self, column: str, value: str, case: bool = True, regex: bool = False) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_contains(data=self.papyrus_bioactivity_data, column=column,
                                                              value=value, case=case, regex=regex),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def not_contains(self, column: str, value: str, case: bool = True, regex: bool = False) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_not_contains(data=self.papyrus_bioactivity_data, column=column,
                                                                  value=value, case=case, regex=regex),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def isin(self, column: str, values: Union[Any, List[Any]]) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_match(data=self.papyrus_bioactivity_data, column=column,
                                                           values=values),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def not_isin(self, column: str, values: Union[Any, List[Any]]) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_not_match(data=self.papyrus_bioactivity_data, column=column,
                                                               values=values),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)


class FPSubSim2Engine:
    """Engine allowing the creation and loading of FPSubSim2 files."""

    def __init__(self, papyrus_params: Dict):
        self.papyrus_params = papyrus_params
        self.path = None
        self.progress = False
        self.fpsubsim2 = subsim_search.FPSubSim2()

    def __call__(self,
                 fingerprint: Optional[Union[
                     subsim_search.Fingerprint, List[subsim_search.Fingerprint]]] = subsim_search.MorganFingerprint(),
                 path: Optional[str] = None,
                 progress: bool = False
                 ) -> FPSubSim2Engine:
        self.fingerprint = fingerprint
        self.path = path
        self.progress = progress
        return self

    def _validate(self):
        # Determine path
        if self.path is None:
            # Get Papyrus path
            if self.papyrus_params['source_path'] is not None:
                os.environ['PYSTOW_HOME'] = os.path.abspath(self.papyrus_params['source_path'])
            # Get Papyrus version
            version = IO.process_data_version(version=self.papyrus_params['version'],
                                              root_folder=self.papyrus_params['source_path'])
            # Determine path in Papyrus folder
            name = (f'{version}_combined_set_with{"out" if not self.papyrus_params["is3d"] else ""}'
                    '_stereochemistry_FPSubSim2.h5')
            self.path = pystow.module('papyrus', version).join(name=name).as_posix()
        # Verify the file exists
        self._exists = os.path.exists(self.path)
        self._valid_parent_folder = os.path.exists(os.path.join(self.path, os.pardir))
        if not self._valid_parent_folder:
            raise NotADirectoryError(f'Cannot create the FPSubSim2 file in a non-existing folder: {self.path}')
        if not self._exists:
            raise FileNotFoundError('The FPSubSim2 file does not exist; consider creating it.')

    def _create(self) -> FPSubSim2Engine:
        self._validate()
        if self._exists:
            self.fpsubsim2.load(fpsubsim_path=self.path)
        else:
            self.fpsubsim2.create_from_papyrus(is3d=self.papyrus_params['is3d'],
                                               version=self.papyrus_params['version'],
                                               outfile=self.path,
                                               fingerprint=self.fingerprint,
                                               root_folder=self.papyrus_params['source_path'],
                                               progress=self.progress)
        return self

    def _load(self) -> FPSubSim2Engine:
        return self._create()

    def _set_data(self,
                  papyrus_bioactivity_data: Union[Iterator[pd.DataFrame], pd.DataFrame],
                  papyrus_protein_data: pd.DataFrame):
        self.papyrus_bioactivity_data = papyrus_bioactivity_data
        self.papyrus_protein_data = papyrus_protein_data

    def keep_similar_molecules(self, smiles: Union[str, List[str]],
                               fingerprint: fingerprint.Fingerprint = fingerprint.MorganFingerprint(),
                               threshold: float = 0.7, cuda: bool = False) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_similar(data=self.papyrus_bioactivity_data,
                                                             molecule_smiles=smiles,
                                                             fpsubsim2_file=self.path,
                                                             fingerprint=fingerprint,
                                                             threshold=threshold,
                                                             cuda=cuda),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def keep_dissimilar_molecules(self, smiles: Union[str, List[str]],
                                  fingerprint: fingerprint.Fingerprint = fingerprint.MorganFingerprint(),
                                  threshold: float = 0.7, cuda: bool = False) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_dissimilar(data=self.papyrus_bioactivity_data,
                                                                molecule_smiles=smiles,
                                                                fpsubsim2_file=self.path,
                                                                fingerprint=fingerprint,
                                                                threshold=threshold,
                                                                cuda=cuda),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def keep_substructure_molecules(self, smiles: Union[str, List[str]]) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_substructure(data=self.papyrus_bioactivity_data,
                                                                  molecule_smiles=smiles,
                                                                  fpsubsim2_file=self.path),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)

    def keep_not_substructure_molecules(self, smiles: Union[str, List[str]]) -> PapyrusDataset:
        return PapyrusDataset._from_data(
            papyrus_bioactivity_data=preprocess.keep_not_substructure(data=self.papyrus_bioactivity_data,
                                                                      molecule_smiles=smiles,
                                                                      fpsubsim2_file=self.path),
            papyrus_protein_data=self.papyrus_protein_data,
            papyrus_params=self.papyrus_params)


class PapyrusMoleculeSet:

    def __init__(self, df: Union[pd.DataFrame, Iterator], papyrus_params: Dict):
        self.data = df
        self.papyrus_params = papyrus_params
        self.num_rows = IO.get_num_rows_in_file(filetype='structures', is3D=self.papyrus_params['is3d'],
                                                version=self.papyrus_params['version'],
                                                plusplus=self.papyrus_params['plusplus'],
                                                root_folder=self.papyrus_params['source_path'])

    def to_dataframe(self, progress: bool = False):
        if isinstance(self.data, Iterator):
            return self.aggregate(progress=progress)
        return self.data

    def __repr__(self):
        if isinstance(self.data, Iterator):
            return f'{type(self).__name__}<iterator of molecules>'
        return f'{type(self).__name__}<{len(self.data)} molecules>'

    def molecular_descriptors(self, desc_type: str, progress: bool = False) -> pd.DataFrame:
        ids = self.aggregate(progress)['connectivity' if self.papyrus_params['is3d'] else 'InChIKey'].unique()
        # Handle descriptors not yet downloaded
        try:
            return reader.read_molecular_descriptors(desc_type=desc_type,
                                                     is3d=self.papyrus_params['is3d'],
                                                     version=self.papyrus_params['version'],
                                                     chunksize=self.papyrus_params['chunksize'],
                                                     source_path=self.papyrus_params['source_path'],
                                                     ids=ids,
                                                     verbose=progress)
        except FileNotFoundError:
            download.download_papyrus(outdir=self.papyrus_params['source_path'],
                                      version=self.papyrus_params['version'],
                                      nostereo=not self.papyrus_params['is3d'], stereo=self.papyrus_params['is3d'],
                                      only_pp=self.papyrus_params['plusplus'], structures=False,
                                      descriptors=desc_type, progress=self.papyrus_params['download_progress'],
                                      disk_margin=0.0)
            return self.molecular_descriptors(desc_type, progress)

    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        total = (-(-self.num_rows // self.papyrus_params['chunksize'])
                 if self.papyrus_params['chunksize'] is not None
                 else None)
        if isinstance(self.data, pd.DataFrame):
            return self.data
        return preprocess.consume_chunks(generator=self.data,
                                         progress=progress, total=total)

    def agg(self, progress: bool = False) -> pd.DataFrame:
        return self.aggregate(progress=progress)

    def consume_chunks(self, progress: bool = False) -> pd.DataFrame:
        return self.aggregate(progress=progress)

class PapyrusProteinSet:
    def __init__(self, df: Union[pd.DataFrame, Iterator], papyrus_params: Dict, num_proteins: int):
        self.data = df
        self.papyrus_params = papyrus_params
        self.num_rows = num_proteins

    def __repr__(self):
        if isinstance(self.data, Iterator):
            return f'{type(self).__name__}<iterator of proteins>'
        return f'{type(self).__name__}<{len(self.data)} proteins>'

    def to_dataframe(self, progress: bool = False) -> pd.DataFrame:
        if isinstance(self.data, Iterator):
            total = (-(-self.num_rows // self.papyrus_params['chunksize'])
                     if self.papyrus_params['chunksize'] is not None
                     else None)
            return preprocess.consume_chunks(generator=self.data, progress=progress, total=total)
        return self.data

    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        return self.to_dataframe(progress)

    def agg(self, progress: bool = False) -> pd.DataFrame:
        return self.to_dataframe(progress=progress)

    def consume_chunks(self, progress: bool = False) -> pd.DataFrame:
        return self.to_dataframe(progress=progress)

class PapyrusPDBProteinSet:

    def __init__(self, df: Union[pd.DataFrame, Iterator], papyrus_params: Dict, num_proteins: int):
        self.data = df
        self.papyrus_params = papyrus_params
        self.num_rows = num_proteins

    def to_dataframe(self, progress: bool = False) -> pd.DataFrame:
        if isinstance(self.data, Iterator):
            total = (-(-self.num_rows // self.papyrus_params['chunksize'])
                     if self.papyrus_params['chunksize'] is not None
                     else None)
            return preprocess.consume_chunks(generator=self.data, progress=progress, total=total)
        return self.data

    def __repr__(self):
        if isinstance(self.data, Iterator):
            return f'{type(self).__name__}<iterator of proteins structures>'
        return f'{type(self).__name__}<{len(self.data)} proteins structures>'

    def protein_descriptors(self,
                            desc_type: Union[str, prodec.Descriptor, prodec.Transform],
                            progress: bool = False
                            ) -> pd.DataFrame:
        ids = self.aggregate(progress)['target_id'].unique()
        try:
            return reader.read_protein_descriptors(desc_type=desc_type,
                                                   is3d=self.papyrus_params['is3d'],
                                                   version=self.papyrus_params['version'],
                                                   chunksize=self.papyrus_params['chunksize'],
                                                   source_path=self.papyrus_params['source_path'],
                                                   ids=ids,
                                                   verbose=progress)
        except FileNotFoundError:
            download.download_papyrus(outdir=self.papyrus_params['source_path'],
                                      version=self.papyrus_params['version'],
                                      nostereo=not self.papyrus_params['is3d'], stereo=self.papyrus_params['is3d'],
                                      only_pp=self.papyrus_params['plusplus'], structures=False,
                                      descriptors=desc_type, progress=self.papyrus_params['download_progress'],
                                      disk_margin=0.0)
            return self.protein_descriptors(desc_type, progress)

    def aggregate(self, progress: bool = False) -> pd.DataFrame:
        return self.to_dataframe(progress)

    def agg(self, progress: bool = False) -> pd.DataFrame:
        return self.to_dataframe(progress=progress)

    def consume_chunks(self, progress: bool = False) -> pd.DataFrame:
        return self.to_dataframe(progress=progress)
