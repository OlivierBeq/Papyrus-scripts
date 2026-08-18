# -*- coding: utf-8 -*-

"""A collection of scripts to handle the Papyrus bioactivity dataset."""

from .download import download_papyrus, remove_papyrus
from .matchRCSB import get_matches, update_rcsb_data
from .modelling import pcm, qsar
from .oop import PapyrusDataset
from .preprocess import (
                     consume_chunks,
                     keep_accession,
                     keep_contains,
                     keep_dissimilar,
                     keep_match,
                     keep_not_contains,
                     keep_not_match,
                     keep_not_substructure,
                     keep_organism,
                     keep_protein_class,
                     keep_quality,
                     keep_similar,
                     keep_source,
                     keep_substructure,
                     keep_type,
                     yscrambling,
)
from .reader import (
                     read_molecular_descriptors,
                     read_molecular_structures,
                     read_papyrus,
                     read_protein_descriptors,
                     read_protein_set,
)
from .utils import IO, UniprotMatch
from .utils.IO import PapyrusVersion
from .utils.mol_reader import MolSupplier

__version__ = '3.0.0'