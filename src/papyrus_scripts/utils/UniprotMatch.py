# -*- coding: utf-8 -*-

"""Functions to interact with UniProt."""

from io import StringIO
from typing import List, Union

import pandas as pd
import requests


def uniprot_mappings(query: Union[str, List[str]],
                     map_from: str = 'ID',
                     map_to: str = 'PDB_ID',
                     ) -> pd.DataFrame:
    """Map identifiers using the UniProt identifier mapping tool.

    :param query: list or space delimited string of identifiers
    :param map_from: type of input identifiers (default: accession)
    :param map_to: type of desired output identifiers
                   (default: PDB identifiers)

    See: https://www.uniprot.org/help/api_idmapping
    """
    url = 'https://www.uniprot.org/uploadlists/'
    if isinstance(query, list):
        query = ' '.join(query)

    params = {'from': map_from,
              'to': map_to,
              'format': 'tab',
              'query': query,
              }

    response = requests.post(url, params)
    if not response.ok:
        raise ValueError("query is wrongly formatted and resulted in a server failure")
    data = StringIO(response.text)
    df = pd.read_csv(data, sep='\t')
    df = df.rename(columns={'To': map_to, 'From': map_from})
    return df
