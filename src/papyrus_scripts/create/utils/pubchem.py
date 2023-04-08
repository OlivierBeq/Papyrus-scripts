# -*- coding: utf-8 -*-

import re
from typing import Iterable, List
import xml.etree.ElementTree as ET

import requests
import pandas as pd


def get_pubchem_assays_depodate(src_assay_ids: Iterable[str]) -> pd.DataFrame:
    """Obtain PubChem assay deposition date from AIDs

    :param src_assay_ids: IDs of PubChem assays
    :return: formatted assay data (AID, deposition year)
    """
    # Contract ids
    src_assay_ids = ','.join(src_assay_ids)
    # Obtain XML data
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{src_assay_ids}/dates/XML?dates_type=deposition"
    xml_response = requests.get(url).text
    # Parse XML
    xroot = ET.fromstring(xml_response)
    rows = []
    for info_node in xroot:
        aid = info_node[0].text
        year = info_node[1][0].text
        rows.append({"src_assay_id": aid, "Year": year})
    return pd.DataFrame(rows)


def map_pubchem_assays(data: pd.DataFrame,
                       col_pubchem_aid: str) -> pd.DataFrame:
    """Add deposition year to the current dataframe.

    :param data: Dataframe to merge data onto.
    :param col_pubchem_aid: Name of the columns containing the AID to map onto
    :return: Merged data
    """
    # Obtain deposition years
    deposition_data = get_pubchem_assays_depodate(data[col_pubchem_aid].unique().tolist())
    # Merge on original data
    return data.merge(deposition_data, on=col_pubchem_aid, how='left')
