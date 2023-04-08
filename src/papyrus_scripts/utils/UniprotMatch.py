# -*- coding: utf-8 -*-

"""Functions to interact with UniProt."""

import re
import json
import time
import zlib
from typing import List, Union
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry


def uniprot_mappings(query: Union[str, List[str]],
                     map_from: str = 'ID',
                     map_to: str = 'PDB_ID',
                     taxon: str = None
                     ) -> pd.DataFrame:
    """Map identifiers using the UniProt identifier mapping tool.

    :param query: list or space delimited string of identifiers
    :param map_from: type of input identifiers (default: accession)
    :param map_to: type of desired output identifiers
                   (default: PDB identifiers)
    :param taxon: taxon to be mapped to if 'map_from' is 'Gene_Name'

    If mapping from {'PDB', 'PDB_ID'} to {'UniProtKB_AC-ID', 'ACC'}
    and query is None, then returns all SIFTS mappings.

    See: https://www.uniprot.org/help/api_idmapping
    """
    if isinstance(query, str):
        query = [query]
    # If mapping PDB to UniProt, use SIFTS flat files
    if map_from in ['PDB', 'PDB_ID'] and map_to in ['UniProtKB_AC-ID', 'ACC']:
        # Obtain mappings from SIFTS
        data = pd.read_csv('ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_pdb.tsv.gz',
                           sep='\t', skiprows=[0]
                 ).rename(columns={'SP_PRIMARY': map_to, 'PDB': map_from})
        # Reorganize columns
        data = data[[map_from, map_to]]
        # Split by PDB
        data[map_from] = data[map_from].str.split(';')
        # Unmerge rows according to PDB
        data = data.explode(column=map_from).reset_index(drop=True)
        if query is not None:
            query = [x.lower() for x in query]
            data = data[data[map_from].str.lower().isin(query)]
        return data
    else:
        # Use UniProt API
        matching = UniprotMatch()
        matches = matching.uniprot_id_mapping(query, map_from, map_to, taxon, verbose=False)
        df = pd.DataFrame.from_dict(matches, orient='index')
        df = df.reset_index().rename(columns={'index': map_from, 0: map_to})
        return df


class UniprotMatch:
    def __init__(self,
                 polling_interval: int = 3,
                 api_url: str = 'https://rest.uniprot.org',
                 retry: Retry = None):
        """Instantiate a class to match UniProt identifiers.

        Based on: https://www.uniprot.org/help/id_mapping#submitting-an-id-mapping-job
        """
        self._api_url = api_url
        self._polling_interval = polling_interval
        if retry is None:
            self._retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        else:
            self._retries = retry
        self._session = requests.Session()
        self._session.mount("https://", HTTPAdapter(max_retries=self._retries))


    def _submit_id_mapping(self, from_db, to_db, ids, taxon=None):
        if from_db == 'Gene_Name' and taxon is None:
            raise ValueError('Taxon must be provided when mapping from gene names.')
        if taxon is None:
            request = requests.post(
                f"{self._api_url}/idmapping/run",
                data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
            )
        else:
            request = requests.post(
                f"{self._api_url}/idmapping/run",
                data={"from": from_db, "to": to_db, "ids": ",".join(ids), "taxId": taxon}
            )
        request.raise_for_status()
        return request.json()["jobId"]

    def _get_next_link(self, headers):
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)

    def _check_id_mapping_results_ready(self, job_id, verbose):
        while True:
            request = self._session.get(f"{self._api_url}/idmapping/status/{job_id}")
            request.raise_for_status()
            j = request.json()
            if "jobStatus" in j:
                if j["jobStatus"] == "RUNNING":
                    if verbose:
                        print(f"Retrying in {self._polling_interval}s")
                    time.sleep(self._polling_interval)
                else:
                    raise Exception(request["jobStatus"])
            else:
                return bool(j["results"] or j["failedIds"])

    def _get_batch(self, batch_response, file_format, compressed):
        batch_url = self._get_next_link(batch_response.headers)
        while batch_url:
            batch_response = self._session.get(batch_url)
            batch_response.raise_for_status()
            yield self._decode_results(batch_response, file_format, compressed)
            batch_url = self._get_next_link(batch_response.headers)

    def _combine_batches(self, all_results, batch_results, file_format):
        if file_format == "json":
            for key in ("results", "failedIds"):
                if key in batch_results and batch_results[key]:
                    all_results[key] += batch_results[key]
        elif file_format == "tsv":
            return all_results + batch_results[1:]
        else:
            return all_results + batch_results
        return all_results

    def _get_id_mapping_results_link(self, job_id):
        url = f"{self._api_url}/idmapping/details/{job_id}"
        request = self._session.get(url)
        request.raise_for_status()
        return request.json()["redirectURL"]

    def _decode_results(self, response, file_format, compressed):
        if compressed:
            decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
            if file_format == "json":
                j = json.loads(decompressed.decode("utf-8"))
                return j
            elif file_format == "tsv":
                return [line for line in decompressed.decode("utf-8").split("\n") if line]
            elif file_format == "xlsx":
                return [decompressed]
            elif file_format == "xml":
                return [decompressed.decode("utf-8")]
            else:
                return decompressed.decode("utf-8")
        elif file_format == "json":
            return response.json()
        elif file_format == "tsv":
            return [line for line in response.text.split("\n") if line]
        elif file_format == "xlsx":
            return [response.content]
        elif file_format == "xml":
            return [response.text]
        return response.text

    def _get_xml_namespace(self, element):
        m = re.match(r"\{(.*)\}", element.tag)
        return m.groups()[0] if m else ""

    def _merge_xml_results(self, xml_results):
        merged_root = ElementTree.fromstring(xml_results[0])
        for result in xml_results[1:]:
            root = ElementTree.fromstring(result)
            for child in root.findall("{http://uniprot.org/uniprot}entry"):
                merged_root.insert(-1, child)
        ElementTree.register_namespace("", self._get_xml_namespace(merged_root[0]))
        return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)

    def _print_progress_batches(self, batch_index, size, total):
        n_fetched = min((batch_index + 1) * size, total)
        print(f"Fetched: {n_fetched} / {total}")

    def _get_id_mapping_results_search(self, url, verbose: bool = False):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        if "size" in query:
            size = int(query["size"][0])
        else:
            size = 500
            query["size"] = size
        compressed = (
            query["compressed"][0].lower() == "true" if "compressed" in query else False
        )
        parsed = parsed._replace(query=urlencode(query, doseq=True))
        url = parsed.geturl()
        request = self._session.get(url)
        request.raise_for_status()
        results = self._decode_results(request, file_format, compressed)
        total = int(request.headers["x-total-results"])
        if verbose:
            self._print_progress_batches(0, size, total)
        for i, batch in enumerate(self._get_batch(request, file_format, compressed), 1):
            results = self._combine_batches(results, batch, file_format)
            if verbose:
                self._print_progress_batches(i, size, total)
        if file_format == "xml":
            return self._merge_xml_results(results)
        return results

    def _get_id_mapping_results_stream(self, url):
        if "/stream/" not in url:
            url = url.replace("/results/", "/stream/")
        request = self._session.get(url)
        request.raise_for_status()
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        compressed = (
            query["compressed"][0].lower() == "true" if "compressed" in query else False
        )
        return self._decode_results(request, file_format, compressed)

    def uniprot_id_mapping(self,
            ids: list, from_db: str = "UniProtKB_AC-ID", to_db: str = None,
            taxon: str = None, verbose: bool = True
    ) -> dict:
        """
        Map Uniprot identifiers into other databases.

		For a list of the available identifiers, check the
        `To database` list on https://www.uniprot.org/id-mapping

        :param ids: IDs to be mapped from
        :param from_db: Type of identifier supplied through 'ids'
        :param to_db: Type of identifier to be obtained
        :param taxon: Taxon ID of the species if 'from_db' is 'Gene_Name'
        :param verbose: Increase verbosity
        :return: A dictionary with query ids as keys and the respective mapped results

        Adapted from David Araripe's (@DavidAraripe) original code
        """
        job_id = self._submit_id_mapping(from_db=from_db, to_db=to_db, ids=ids, taxon=taxon)
        if self._check_id_mapping_results_ready(job_id, verbose):
            link = self._get_id_mapping_results_link(job_id)
            r = self._get_id_mapping_results_search(link)
            r_dict = {idx: r["results"][idx] for idx in range(len(r["results"]))}
            r_df = pd.DataFrame.from_dict(r_dict, orient="index")
            query_to_newIDs = dict()
            for id in r_df["from"].unique():
                subset_df = r_df[r_df["from"] == id]
                if isinstance(subset_df["to"].tolist()[0], str):
                    query_to_newIDs[id] = " ".join(list(subset_df["to"].unique()))
                elif isinstance(subset_df["to"].tolist()[0], dict):
                    query_to_newIDs[id] = " ".join(set(subset_df["to"].apply(lambda row: row['primaryAccession'])))
            return query_to_newIDs
