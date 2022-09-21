# -*- coding: utf-8 -*-

import pandas as pd
from tqdm.auto import tqdm

from natsort import natsorted
from jax_unirep import get_reps
from jax_unirep.utils import load_params


def get_unirep_descriptors(protein_data: pd.DataFrame,
                           col_seq_id: str,
                           col_sequence: str,
                           progress: bool = True) -> pd.DataFrame:
    """Obtain UniRep embeddings of proteins.

    :param protein_data: A dataframe with protein sequences and identifiers
    :param col_seq_id: Name of the columns containing protein identifiers
    :param col_sequence: Name of the columns containing protein sequences
    :param progress: Whether to show progress
    """
    unirep_sizes = [64, 256, 1900]
    data = []
    if progress:
        pbar = tqdm(protein_data.iterrows(),
                    total=protein_data.shape[0],
                    desc='Obtaining UniRep embeddings')
    else:
        pbar = protein_data.iterrows()
    # Iterate over sequences
    for i, row in pbar:
        # Obtain hidden average, final hidden and final cell states for each ambedding size
        # Organize so that all hidden average, final hidden and final cell states are grouped
        embeddings = list(zip(*[get_reps(row[col_sequence],
                                         params=load_params(paper_weights=uni_size)[1],
                                         mlstm_size=uni_size)
                                for uni_size in unirep_sizes]))
        # Organize in a dict
        values = {col_seq_id: row[col_seq_id],
                  **{'UniRep%s_AH_%s' % (size, k+1): x for j, size in enumerate(unirep_sizes) for k, x in enumerate(embeddings[0][j][0])},
                  **{'UniRep%s_FH_%s' % (size, k+1): x for j, size in enumerate(unirep_sizes) for k, x in enumerate(embeddings[1][j][0])},
                  **{'UniRep%s_FC_%s' % (size, k+1): x for j, size in enumerate(unirep_sizes) for k, x in enumerate(embeddings[2][j][0])}}

        data.append(values)

    # To pandas
    data = pd.DataFrame(data)

    # Reorder (should dict not be sorted anymore)
    data = data[[col_seq_id] + [natsorted(data.drop(columns=[col_seq_id]).columns)]]

    return data
