# -*- coding: utf-8 -*-

"""A collection of scripts to handle the Papyrus bioactivity dataset."""

from .download import download_papyrus, remove_papyrus
from .reader import (read_papyrus, read_protein_set, read_protein_descriptors,
                     read_molecular_descriptors, read_molecular_structures)

from .matchRCSB import update_rcsb_data, get_matches
from .preprocess import (keep_organism, keep_accession, keep_type, keep_source,
                         keep_protein_class, keep_quality, keep_contains, keep_match,
                         keep_similar, keep_substructure, consume_chunks, yscrambling)

from .modelling import qsar, pcm

from .utils.mol_reader import MolSupplier
from .utils import IO, UniprotMatch

__version__ = '1.0.2'
