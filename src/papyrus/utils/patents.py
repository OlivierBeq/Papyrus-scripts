# -*- coding: utf-8 -*-

import re
import warnings
import textwrap
from typing import Iterable, List

import pandas as pd


def process_patent_ids(patent_ids: Iterable[str]) -> List[str]:
    """Format the patent IDs

    :param patent_ids: IDs of patents
    :return: formatted patent IDs
    """
    ids = []
    for id in patent_ids:
        if len(re.search(r'(.{2})-(\d+)-([A-Z]\d?)', id).groups()) == 3:
            ids.append(id)
        else:
            values = re.search(r'(.{2})(\d+)([A-Z]\d?)', id).groups()
            if len(values) == 3:
                ids.append(f'{values[0]}-{values[1]}-{values[2]}')
    return ids


def create_big_query_request(patent_ids: List[str]) -> str:
    """Create a request to submit to the Google BigQuery API to
    obtain patent details.

    :param patent_ids: IDs of patents
    :return: query to bumity to Google API
    """
    patent_ids = process_patent_ids(set(patent_ids))
    return f"""SELECT DISTINCT country_code, 
publication_number, 
application_number, 
pct_number, 
family_id, 
publication_date, 
filing_date, 
priority_date 
FROM `patents-public-data.patents.publications` 
WHERE publication_number LIKE "{'" OR publication_number LIKE "'.join(patent_ids)}";"""

def map_patent_id(data: pd.DataFrame,
                  bigquery_data: pd.DataFrame,
                  manually_mapped_data: pd.DataFrame,
                  col_patent_id: str,
                  warning: str = 'raise') -> pd.DataFrame:
    """Merge patent data manually obtained  or with Google BigQuery API to
    the supplied dataframe.

    :param data: Dataframe to merge data onto.
    :param bigquery_data: Data from Google BigQuery
    :param manually_mapped_data: Manually mapped patent data
    :param col_patent_id: Name of the columns containing the patent ID to map onto
    :param warnings: How should warning s be reported {'raise', 'warn'}.
                     If 'raise', an exception is raised, other a warning is issued.
    :return: Merged data
    """
    # Raise or warn upon missing patent data
    unmapped = data[~data[col_patent_id].isin(bigquery_data.publication_number) & ~data[col_patent_id].isin(manually_mapped_data.publication_number)]
    if not unmapped.empty:
        out = ('patent publication number was missing for the following patents:\n' +
               textwrap.fill(', '.join(unmapped[col_patent_id].tolist())) + '\n')
        if warning == 'raise':
            raise AttributeError(out)
        else:
            warnings.warn(out, category=RuntimeWarning)
    # Map  patents
    bigquery = data.rename(columns={col_patent_id: 'publication_number'})\
                   .merge(bigquery_data, on='publication_number')
    bigquery['year'] = bigquery.filing_date.str.slice(0,5)
    uspto = bigquery.merge(manually_mapped_data, on='publication_number')
    return pd.concat([bigquery.rename(columns={'publication_number': col_patent_id})
                              .reset_index(drop=True),
                      uspto.rename(columns={'publication_number': col_patent_id})
                           .reset_index(drop=True)
                      ]).reset_index(drop=True)