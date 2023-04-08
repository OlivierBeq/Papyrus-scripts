# -*- coding: utf-8 -*-

"""ChEMBL data mining."""

import os
import re
from typing import Optional
from functools import partial
from itertools import chain

import pystow
import chembl_downloader
import pandas as pd
from rdkit import Chem
from pandas.io.parsers import TextFileReader as PandasTextFileReader

from .utils.pubchem import map_pubchem_assays
from .utils.patents import map_patent_id
from .utils.uniprot import uniprot_information
from .utils.pandas_utils import equalize_cell_size_in_row
from .utils.mol_standardize import standardize
from .utils.utils import process_and_write_chunks, remove_temporary_files

BASE_DIR = os.path.dirname(__file__)

def process_chembl_data(papyrus_version: str,
                        chembl_version: Optional[int] = None,
                        prefix: Optional[str] = 'papyrus-creation',
                        include_variants: bool = False,
                        chunksize: int = 10_000) -> None:
    """Process data from ChEMBL to be integrated in Papyrus

    :param papyrus_version: The version of the Papyrus dataset being created.
    :param chembl_version: Specific version of ChEMBL to be considered (default: None -> latest)
    :param prefix: Prefix directory, in pystow's data folder, where intermediary files
                   and ChEMBL SQLite will be stored.
    :param include_variants: Whether to include variant information (must be improved before can be used)
    :param chunksize: maximum number of entries to be processed at once
    """
    # Determine default paths
    papyruslib_root = prefix
    chembl_root = 'chembl'

    # Ensure no temporary file exist in the current folder
    remove_temporary_files(papyruslib_root, chembl_root)

    # chembl-downloader handles the download of the SQLite db if need be

    # Activities
    activity_query = """
    SELECT activities.activity_id,
           activities.molregno,
           activities.assay_id,
           activities.pchembl_value,
           activities.potential_duplicate,
           activities.data_validity_comment,
           activities.activity_comment,
           activities.standard_relation,
           activities.standard_value,
           activities.standard_units,
           activities.standard_flag,
           activities.standard_type,
           activities.doc_id,
           activities.record_id,
           activities.src_id
    FROM activities
    """
    activity_df: PandasTextFileReader = chembl_downloader.query(activity_query, version=chembl_version,
                                                                prefix=[papyruslib_root, chembl_root],
                                                                chunksize=chunksize)

    def process_protential_duplicates_activities(activity_df: pd.DataFrame, negate:bool = False):
        # Remove duplicates
        mask = activity_df.potential_duplicate == 1
        pystow.dump_df(papyruslib_root, chembl_root,
                       name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_potential_duplicates.txt',
                       obj=activity_df[mask])
        activity_df  = activity_df[~mask]
        # Remove invalid data
        mask = activity_df.data_validity_comment.isin(['Outside typical range', 'Non standard unit for type',
                                                       'Potential missing data', 'Potential transcription error',
                                                       'Potential author error', 'Author confirmed error'])
        questionned = activity_df[mask]
        submask = questionned.data_validity_comment != 'Author confirmed error' # Force removal of confirmed errors
        if not negate:
            # Return potential duplicates
            return questionned[submask]
        else:
            # Return all but potential duplicates
            return activity_df[~mask]

    # Write questionned bioactivities
    process_and_write_chunks(activity_df,
                             process_protential_duplicates_activities,
                             f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_questionned_valid_activity.txt',
                             papyruslib_root, chembl_root)

    def process_censored_activities(activity_df: pd.DataFrame, negate: bool = False):
        activity_df = process_protential_duplicates_activities(activity_df, negate=True)
        # Keep measures with '=' sign
        mask = activity_df.standard_relation != '='
        if not negate:
            # Return censored data
            return activity_df[mask]
        else:
            # Return all other data
            return activity_df[~mask]

    # Iterative reader is exhausted, need to reinstantiate
    activity_df: PandasTextFileReader = chembl_downloader.query(activity_query, version=chembl_version,
                                                                prefix=[papyruslib_root, chembl_root],
                                                                chunksize=chunksize)
    # Write censored bioactivities
    process_and_write_chunks(activity_df,
                             process_censored_activities,
                             f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_censored_activity.txt',
                             papyruslib_root,
                             chembl_root)

    def process_undefined_activities(activity_df: pd.DataFrame, negate: bool = False):
        activity_df = process_censored_activities(activity_df, negate=True)
        # pChEMBL value is not defined
        mask = activity_df.pchemb_value.isna()
        if not negate:
            # Return binary values
            return activity_df[mask]
        else:
            # Return other values
            return activity_df[~mask]

    # Iterative reader is exhausted, need to reinstantiate
    activity_df: PandasTextFileReader = chembl_downloader.query(activity_query, version=chembl_version,
                                                                prefix=[papyruslib_root, chembl_root],
                                                                chunksize=chunksize)
    # Write undefined bioactivities
    process_and_write_chunks(activity_df,
                             process_undefined_activities,
                             f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_undefined_pchembl_values.txt',
                             papyruslib_root,
                             chembl_root)

    def process_inconclusive_activities(activity_df: pd.DataFrame, negate: bool = False):
        activity_df = process_undefined_activities(activity_df, negate=True)
        # Is activity questioned?
        mask = activity_df.activity_comment.isin(['inconclusive', 'unspecified', 'Indeterminate',
                                                  'Ineffective', 'Insoluble', 'Insufficient',
                                                  'Lack of solubility', 'Inconclusive',
                                                  'Not Determined', 'ND(Insoluble)', 'tde',
                                                  'insoluble', 'not tested', 'uncertain',
                                                  'No compound available', 'No compound detectable',
                                                  'No data', 'Non valid test', 'Not assayed',
                                                  'OUTCOME = Not detected', 'Precipitate',
                                                  'Precipitated', 'Precipitation',
                                                  'Precipitates under the conditions of the study',
                                                  'Qualitative measurement', 'Too insoluble',
                                                  'Unable to be measured', 'Unable to calculate',
                                                  'Uncertain'])
        if not negate:
            # Return questionned activities
            return activity_df[mask]
        else:
            # Return other activities
            return activity_df[~mask]

    # Iterative reader is exhausted, need to reinstantiate
    activity_df: PandasTextFileReader = chembl_downloader.query(activity_query, version=chembl_version,
                                                                prefix=[papyruslib_root, chembl_root],
                                                                chunksize=chunksize)
    # Write inconclusive bioactivities
    process_and_write_chunks(activity_df,
                             process_inconclusive_activities,
                             f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_inconclusive_activity.txt',
                             papyruslib_root, chembl_root)

    def process_pubchem_activities(activity_df: pd.DataFrame, negate: bool = False):
        activity_df = process_inconclusive_activities(activity_df, negate=True)
        # PubChem origin src_id = 7
        mask = activity_df.src_id == 7
        if not negate:
            return activity_df[mask]
        else:
            return activity_df[~mask]

    # Iterative reader is exhausted, need to reinstantiate
    activity_df: PandasTextFileReader = chembl_downloader.query(activity_query, version=chembl_version,
                                                                prefix=[papyruslib_root, chembl_root],
                                                                chunksize=chunksize)
    # Write PubChem bioactivities
    process_and_write_chunks(activity_df,
                             process_pubchem_activities,
                             f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_pubchem_bioassays.txt',
                             papyruslib_root, chembl_root)

    # Iterative reader is exhausted, need to reinstantiate
    activity_df: PandasTextFileReader = chembl_downloader.query(activity_query, version=chembl_version,
                                                                prefix=[papyruslib_root, chembl_root],
                                                                chunksize=chunksize)
    # Write ChEMBL bioactivities
    process_and_write_chunks(activity_df,
                             partial(process_pubchem_activities, nagate=True),
                             f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)
    del activity_df

    # Compounds
    compound_query = """
    SELECT molecule_hierarchy.molregno,
           molecule_dictionary.pref_name,
           molecule_dictionary.chembl_id,
           molecule_dictionary.max_phase,
           molecule_dictionary.therapeutic_flag,
           molecule_dictionary.dosed_ingredient,
           molecule_dictionary.molecule_type,
           molecule_dictionary.structure_type,
           molecule_dictionary.withdrawn_flag,
           molecule_dictionary.withdrawn_year,
           molecule_dictionary.withdrawn_country,
           molecule_dictionary.withdrawn_reason,
           molecule_dictionary.withdrawn_class
    FROM molecule_hierarchy,
         molecule_dictionary
    WHERE molecule_hierarchy.parent_molregno = molecule_dictionary.molregno
    """
    compounds_df = chembl_downloader.query(compound_query, version=chembl_version,
                                          prefix=[papyruslib_root, chembl_root],
                                           chunksize=chunksize)

    def process_conpounds(compounds_df: pd.DataFrame):
        # Rename chembl_id to compound_chembl_id, ...
        compounds_df.rename(columns={'chembl_id': 'compound_chembl_id',
                                     'pref_name': 'compound_pref_name'},
                            inplace=True)
        # Unique molregno
        compounds_df.drop_duplicates(subset='molregno', inplace=True)
        return compounds_df

    # Write compounds list
    process_and_write_chunks(compounds_df,
                             process_conpounds,
                             f'{papyrus_version}_raw_molecule_list_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)
    del compounds_df

    def merge_chembl_and_pubchem_bioactivities_and_molecules(activity_df):
        # Load compounds
        compounds_df = pystow.load_df(papyruslib_root, chembl_root,
                                      name=f'{papyrus_version}_raw_molecule_list_c{chembl_version}.txt',
                                      read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
        for chunk in compounds_df:
            # Merge ChEMBL & PubChem bioactivities onto molecules
            activity_df = activity_df.merge(chunk, on='molregno')
            # Keep small molecules # TODO: create custom filter as most new molecules are not annotated yet
            activity_df = activity_df[activity_df.molecule_type.str.contains('small molecule')]
            yield activity_df

    # Chain PubChem and ChEMBL bioactivities
    activity_df = chain(pystow.load_df(papyruslib_root, chembl_root,
                                       name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}.txt',
                                       read_csv_kwargs={'sep': '\t', 'chunksize': chunksize}),
                        pystow.load_df(papyruslib_root, chembl_root,
                                       name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_pubchem_bioassays.txt',
                                       read_csv_kwargs={'sep': '\t', 'chunksize': chunksize}))

    process_and_write_chunks(activity_df,
                             merge_chembl_and_pubchem_bioactivities_and_molecules,
                             f'{papyrus_version}_raw_bioactivities_only_small_molecules_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)

    def merge_questionned_bioactivities_and_molecules(activity_df):
        # Load compounds
        compounds_df = pystow.load_df(papyruslib_root, chembl_root,
                                      name=f'{papyrus_version}_raw_molecule_list_c{chembl_version}.txt',
                                      read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
        for chunk in compounds_df:
            # Merge ChEMBL & PubChem bioactivities onto molecules
            activity_df = activity_df.merge(chunk, on='molregno')
            # Keep small molecules # TODO: create custom filter as most new molecules are not annotated yet
            activity_df = activity_df[activity_df.molecule_type.str.contains('small molecule')]
            yield activity_df


    # Chain questionned bioactivities
    activity_df = chain(pystow.load_df(papyruslib_root, chembl_root,
                                       name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_potential_duplicates.txt',
                                       read_csv_kwargs={'sep': '\t', 'chunksize': chunksize}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_questionned_valid_activity.txt',
                                            read_csv_kwargs={'sep': '\t', 'chunksize': chunksize}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_censored_activity.txt',
                                            read_csv_kwargs={'sep': '\t', 'chunksize': chunksize}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_undefined_pchembl_values.txt',
                                            read_csv_kwargs={'sep': '\t', 'chunksize': chunksize}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_inconclusive_activity.txt',
                                            read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
                             )

    process_and_write_chunks(activity_df,
                             merge_questionned_bioactivities_and_molecules,
                             f'{papyrus_version}_low_quality_raw_bioactivities_only_small_molecules_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)
    del activity_df

    # Assays
    assay_query = """
    SELECT assays.assay_id,
           assays.description,
           assays.assay_type,
           assays.assay_test_type,
           assays.relationship_type,
           assays.tid,
           assays.confidence_score,
           assays.chembl_id,
           assays.src_assay_id,
           docs.pubmed_id,
           docs.doi,
           docs.patent_id,
           docs.title,
           docs.chembl_id,
           docs.year,
           assays.variant_id
    FROM assays,
         docs
    WHERE assays.doc_id = docs.doc_id
    """
    assay_df: PandasTextFileReader = chembl_downloader.query(assay_query, version=chembl_version,
                                                             prefix=[papyruslib_root, chembl_root],
                                                             chunksize=chunksize)

    def process_assay_metadata(assay_df, negate: bool = False):
        # Separate AID, doc_id and year
        filter_1 = (~assay_df.year.isna() & ~assay_df.pubmed_id.isna())
        filter_2 = (~assay_df.year.isna() & assay_df.pubmed_id.isna() & ~assay_df.doi.isna())
        filter_3 = (~assay_df.year.isna() & assay_df.pubmed_id.isna() & assay_df.doi.isna() & ~assay_df.patent_id.isna())
        filter_4 = (~assay_df.year.isna() & assay_df.pubmed_id.isna() & assay_df.doi.isna() & assay_df.patent_id.isna())
        filter_5 = (assay_df.year.isna() & (assay_df.chembl_id_1 == "CHEMBL1201862"))  # PubChem Bioassay
        filter_6 = (assay_df.year.isna() & (assay_df.chembl_id_1 == "CHEMBL1909046"))  # DrugMatrix
        filter_7 = (assay_df.year.isna() & ~assay_df.chembl_id_1.isin(["CHEMBL1201862", "CHEMBL1909046"]) & ~assay_df.patent_id.isna())
        filter_8 = (assay_df.year.isna() & ~assay_df.chembl_id_1.isin(["CHEMBL1201862", "CHEMBL1909046"]) & assay_df.patent_id.isna() & assay_df.title.str.contains(r'\(\d{4}\)\s+PMID:'))
        filter_9 = (assay_df.year.isna() & ~assay_df.chembl_id_1.isin(["CHEMBL1201862", "CHEMBL1909046"]) & assay_df.patent_id.isna() & ~assay_df.title.str.contains(r'\(\d{4}\)\s+PMID:'))

        assay_df_1 =assay_df[filter_1]
        assay_df.drop(index=assay_df[~filter_1].index, inplace=True)
        assay_df_1['AID'] = assay_df_1.chembl_id
        assay_df_1.assign(doc_id=['PMID:' + x for x in assay_df_1.pubmed_id])

        assay_df_2 = assay_df[filter_2]
        assay_df.drop(index=assay_df[~filter_2].index, inplace=True)
        assay_df_2['AID'] = assay_df_2.chembl_id
        assay_df_2.assign(doc_id=['DOI:' + x for x in assay_df_2.doi])

        assay_df_3 = assay_df[filter_3]
        assay_df.drop(index=assay_df[~filter_3].index, inplace=True)
        assay_df_3['AID'] = assay_df_3.chembl_id
        assay_df_3.assign(doc_id=['PATENT:' + re.sub('[^a-zA-Z0-9]', '', x)  for x in assay_df_3.patent_id])

        assay_df_4 = assay_df[filter_4]
        assay_df.drop(index=assay_df[~filter_4].index, inplace=True)
        assay_df_4['AID'] = assay_df_4.chembl_id
        assay_df_4['doc_id'] = assay_df_4.chembl_id_1

        assay_df_5 = assay_df[filter_5]
        assay_df.drop(index=assay_df[~filter_5].index, inplace=True)
        assay_df_5 = map_pubchem_assays(assay_df_5, 'src_assay_id')
        assay_df_5['AID'] = assay_df_5.chembl_id
        assay_df_5.assign(doc_id=['PubChemAID:' + x for x in assay_df_5.src_assay_id])

        assay_df_6 = assay_df[filter_6]
        assay_df.drop(index=assay_df[~filter_6].index, inplace=True)
        assay_df_6['year'] = ""
        assay_df_6['AID'] = assay_df_6.chembl_id
        assay_df_6.assign(doc_id=['DrugMatrix:' + x for x in assay_df_6.chembl_id_1])

        assay_df_7 = assay_df[filter_7]
        assay_df.drop(index=assay_df[~filter_7].index, inplace=True)
        assay_df_7 = map_patent_id(assay_df_7,
                                   pd.read_csv(os.path.join(BASE_DIR, 'data', 'bigquery_patents.tsv'), sep='\t'),
                                   pd.read_csv(os.path.join(BASE_DIR, 'data', 'uspto_patents.tsv'), sep='\t'),
                                   'src_assay_id')
        assay_df_7['AID'] = assay_df_7.chembl_id
        assay_df_7.assign(doc_id=['PATENT:' + re.sub('[^a-zA-Z0-9]', '', x) for x in assay_df_7.patent_id])

        assay_df_8 = assay_df[filter_8]
        assay_df.drop(index=assay_df[~filter_8].index, inplace=True)
        assay_df_8 = map_patent_id(assay_df_8,
                                   pd.read_csv(os.path.join(BASE_DIR, 'data', 'bigquery_patents.tsv'), sep='\t'),
                                   pd.read_csv(os.path.join(BASE_DIR, 'data', 'uspto_patents.tsv'), sep='\t'),
                                   'src_assay_id')
        assay_df_8['AID'] = assay_df_8.chembl_id
        assay_df_8.assign(year=[re.sub(r'\((\d{4})\)\s+PMID:\s*\d+', r'\1', x) for x in assay_df_8.title])
        assay_df_8.assign(doc_id=['PMID:' + re.sub(r'\(\d{4}\)\s+PMID:\s*(\d+)', r'\1', x) for x in assay_df_8.title])

        assay_df_9 = assay_df[filter_9]
        assay_df.drop(index=assay_df[~filter_9].index, inplace=True)
        assay_df_9['AID'] = assay_df_9.chembl_id
        assay_df_9['year'] = ""
        assay_df_9['doc_id'] = assay_df_9.chembl_id_1

        data = pd.concat([assay_df_1, assay_df_2, assay_df_3, assay_df_4,
                          assay_df_5, assay_df_6, assay_df_7, assay_df_8,
                          assay_df_9])
        data = data[['assay_id', 'description', 'assay_type', 'assay_test_type', 'relationship_type', 'tid',
                     'confidence_score', 'chembl_id', 'src_assay_id', 'AID', 'doc_id', 'year', 'variant_id']]
        data.rename(columns={'chembl_id': 'assay_chembl_id', 'year': 'Year'}, inplace=True)
        data.drop_duplicates('assay_id', inplace=True)

        if not assay_df.empty:
            raise RuntimeError(f'Some assays were not considered by the filters:\n{assay_df}')

        return data

    process_and_write_chunks(assay_df,
                             process_assay_metadata,
                             f'{papyrus_version}_raw_assay_list_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)

    def merge_bioactivities_and_assays(activity_df: pd.DataFrame, negate: bool = False):
        # Load assays
        assay_df = pystow.load_df(papyruslib_root, chembl_root,
                                  name=f'{papyrus_version}_raw_assay_list_c{chembl_version}.txt',
                                  read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
        for chunk in assay_df:
            # Merge bioactivities and assays
            activity_df.drop(columns=['doc_id'], inplace=True)
            activity_df = activity_df.merge(chunk, on='assay_id')

            # Filter on confidence score
            ## 5   Multiple direct protein targets may be assigned
            ## 7   Direct protein complex subunits assigned
            ## 8   Homologous single protein target assigned
            ## 9   Direct single protein target assigned
            activity_df = activity_df[~activity_df.confidence_score.isna()]

            if not negate:
                # Return high confidence data
                yield activity_df[activity_df.confidence_score.astype(int).isin([9, 8, 7, 5])]
            else:
                # Return lower confidence data
                yield activity_df[~activity_df.confidence_score.astype(int).isin([9, 8, 7, 5])]

    activity_df = pystow.load_df(papyruslib_root, chembl_root,
                                 name=f'{papyrus_version}_raw_bioactivities_only_small_molecules_c{chembl_version}.txt',
                                 read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})

    # Merge bioactivities and assays of higher quality
    process_and_write_chunks(activity_df,
                             merge_bioactivities_and_assays,
                             f'{papyrus_version}_raw_bioactivities_only_small_molecules_conf_9_8_7_5_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)

    # Merge bioactivities and assays of lower quality
    #  - 1: higher quality bioactivities of assays of lower confidence
    activity_df = pystow.load_df(papyruslib_root, chembl_root,
                                 name=f'{papyrus_version}_raw_bioactivities_only_small_molecules_c{chembl_version}.txt',
                                 read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
    #  - 2: lower quality bioactivities
    low_qual_activity_df: PandasTextFileReader = pystow.load_df(papyruslib_root, chembl_root,
                                                                name=f'{papyrus_version}_low_quality_raw_bioactivities'
                                                                     f'_only_small_molecules_c{chembl_version}.txt',
                                                                read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})

    def concat_low_quality_bioactivities_and_assays(low_qual_activity_df: pd.DataFrame, activity_df: pd.DataFrame):
        # Work with higher quality bioactivities of assays of lower confidence
        for chunk in merge_bioactivities_and_assays(activity_df, negate=True):
            yield chunk
        # Load assays
        assay_df = pystow.load_df(papyruslib_root, chembl_root,
                                  name=f'{papyrus_version}_raw_assay_list_c{chembl_version}.txt',
                                  read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
        # Focus on lower quality bioactivities
        for chunk in assay_df:
            low_qual_activity_df.drop(columns=['doc_id'], inplace=True)
            low_qual_activity_df = low_qual_activity_df.merge(chunk, on='assay_id')
            # Keep all confidence scores as input bioactivities are of lower quality
            yield low_qual_activity_df

    process_and_write_chunks(low_qual_activity_df,
                             partial(concat_low_quality_bioactivities_and_assays, activity_df=activity_df),
                             f'{papyrus_version}_raw_bioactivities_only_small_molecules_conf_9_8_7_5_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)

    del activity_df, low_qual_activity_df

    # Protein classifications
    target_query = """
    SELECT target_dictionary.target_type,
           target_dictionary.pref_name,
           target_dictionary.tid,
           target_dictionary.chembl_id
    FROM target_dictionary
    """
    targets: PandasTextFileReader = chembl_downloader.query(target_query, version=chembl_version,
                                                            prefix=[papyruslib_root, chembl_root],
                                                            chunksize=chunksize)

    def process_targets(targets: pd.DataFrame):
        targets.rename(columns={'chembl_id': 'target_chembl_id',
                                'pref_name': 'target_pref_name'},
                       inplace=True)
        target_ids = pd.concat([chembl_downloader.query(f"""
        SELECT target_components.tid,
               component_class.protein_class_id,
               component_sequences.accession,
               component_sequences.component_id
        FROM target_components,
             component_sequences,
             component_class
        WHERE target_components.component_id = component_sequences.component_id
            AND component_sequences.component_id = component_class.component_id
            AND target_components.tid = {tid}""", prefix=[papyruslib_root, chembl_root])
                                    for tid in targets.tid])
        classification = pd.concat([chembl_downloader.query(f"""
        SELECT protein_family_classification.protein_class_id,
               protein_family_classification.l1,
               protein_family_classification.l2,
               protein_family_classification.l3,
               protein_family_classification.l4,
               protein_family_classification.l5,
               protein_family_classification.l6,
               protein_family_classification.l7,
               protein_family_classification.l8
        FROM protein_family_classification
        WHERE protein_family_classification.protein_class_id = {class_id}""", prefix=[papyruslib_root, chembl_root])
                                    for class_id in target_ids.protein_class_id])
        # Format classification
        classification['Classification'] = classification[['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8']].apply(lambda x: x.str.cat(sep='->'), axis=1)
        classification.drop(columns=['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8'], inplace=True)
        # Merge
        targets = targets.merge(target_ids, on='tid', how='outer').merge(classification, on='protein_class_id', how='outer')
        del target_ids, classification
        # Drop if no accession
        targets = targets[~targets.accession.isna()]
        targets = targets.drop_duplicates(subset=['accession', 'Classification'])
        list_uniques = lambda x: ';'.join(set(str(k) for k in x))
        # Keep unique values
        targets = targets.groupby('accession').aggregate({'target_type': list_uniques, 'target_pref_name': list_uniques,
                                                          'tid': list_uniques, 'target_chembl_id': list_uniques,
                                                          'protein_class_id': list_uniques, 'component_id': list_uniques,
                                                          'Classification': list_uniques}
                                                         ).reset_index()
        # Entry Q8MMZ4 is obsolete and has been merged with W7JX98
        targets[targets.accession == 'Q8MMZ4', 'accession'] = 'W7JX98'
        # Remove nucleic acids
        targets = targets[~targets.accession.str.startswith('ENSG')]
        # Obtain UniProt information
        target_data = uniprot_information(targets.accession.tolist())
        targets = targets.merge(target_data, on='accession', how='inner')
        del target_data
        # Explode data
        targets = (targets.set_index('accession')
                          .apply(lambda x: x.str.split(';'))
                          .apply(equalize_cell_size_in_row, axis=1)
                          .apply(pd.Series.explode)
                          .reset_index())
        return targets

    process_and_write_chunks(targets,
                             process_targets,
                             f'{papyrus_version}_raw_target_list_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)
    del targets

    # Variants
    if include_variants:
        # TODO: improve filtering and quality check of variants
        variants: PandasTextFileReader = pystow.load_df(papyruslib_root, chembl_root,
                                                        name=f'{papyrus_version}_raw_assay_list_c{chembl_version}.txt',
                                                        read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})

        def process_variants(variants: pd.DataFrame):
            variants = variants[['variant_id', 'tid']]
            variants = variants[~variants.variant_id.isna()]
            variants.drop_duplicates('variant_id', inplace=True)

            mutations = pd.concat([chembl_downloader.query(f"""
                SELECT variant_sequences.variant_id,
                       variant_sequences.mutation,
                       variant_sequences.accession,
                       variant_sequences.sequence
                FROM variant_sequences
                WHERE variant_sequences.variant_id = {variant_id}""", prefix=[papyruslib_root, chembl_root])
                                   for variant_id in variants.variant_id])
            mutations['Length'] = mutations.sequence.str.len()
            mutations = mutations.sort_values('variant_id')

            targets = pd.concat([chembl_downloader.query(f"""
                    SELECT target_dictionary.target_type,
                           target_dictionary.pref_name,
                           target_dictionary.tid,
                           target_dictionary.chembl_id
                    FROM target_dictionary
                    WHERE target_dictionary.tid = {tid}""", prefix=[papyruslib_root, chembl_root])
                                   for tid in variants.tid])
            targets.rename(columns={'chembl_id': 'target_chembl_id',
                                    'pref_name': 'target_pref_name'},
                           inplace=True)

            variants = variants.merge(mutations, on='variant_id', how='inner').merge(targets, on='tid', how='inner')
            del mutations, targets
            target_ids = pd.concat([chembl_downloader.query(f"""
                SELECT target_components.tid,
                       component_class.protein_class_id,
                       component_sequences.accession,
                       component_sequences.component_id
                FROM target_components,
                     component_sequences,
                     component_class
                WHERE target_components.component_id = component_sequences.component_id
                    AND component_sequences.component_id = component_class.component_id
                    AND target_components.tid = {tid}""", prefix=[papyruslib_root, chembl_root])
                                    for tid in variants.tid])
            classification = pd.concat([chembl_downloader.query(f"""
                SELECT protein_family_classification.protein_class_id,
                       protein_family_classification.l1,
                       protein_family_classification.l2,
                       protein_family_classification.l3,
                       protein_family_classification.l4,
                       protein_family_classification.l5,
                       protein_family_classification.l6,
                       protein_family_classification.l7,
                       protein_family_classification.l8
                FROM protein_family_classification
                WHERE protein_family_classification.protein_class_id = {class_id}""", prefix=[papyruslib_root, chembl_root])
                                        for class_id in target_ids.protein_class_id])
            # Format classification
            classification['Classification'] = classification[['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8']].apply(
                lambda x: x.str.cat(sep='->'), axis=1)
            classification.drop(columns=['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8'], inplace=True)
            # Merge
            variants = variants.merge(target_ids, on='tid', how='outer').merge(classification, on='protein_class_id',
                                                                             how='outer')
            del target_ids, classification
            # Drop if no accession
            variants = variants[~variants.accession.isna()]
            list_uniques = lambda x: ';'.join(set(str(k) for k in x))
            # Keep unique values
            variants = variants.groupby(['accession', 'tid']).aggregate({'target_type': list_uniques, 'target_pref_name': list_uniques,
                                                              'tid': list_uniques, 'target_chembl_id': list_uniques,
                                                              'protein_class_id': list_uniques,
                                                              'component_id': list_uniques,
                                                              'Classification': list_uniques}
                                                         ).reset_index()
            # Entry Q8MMZ4 is obsolete and has been merged with W7JX98
            variants[variants.accession == 'Q8MMZ4', 'accession'] = 'W7JX98'
            # Remove nucleic acids
            variants = variants[~variants.accession.str.startswith('ENSG')]
            # Obtain UniProt information
            variants_data = uniprot_information(variants.accession.tolist())
            variants = variants.merge(variants_data, on='accession', how='inner')
            del variants_data
            # Explode data
            variants = (variants.set_index('accession')
                        .apply(lambda x: x.str.split(';'))
                        .apply(equalize_cell_size_in_row, axis=1)
                        .apply(pd.Series.explode)
                        .reset_index())
            # Sort and keep first occurence
            variants = variants.sort_values('variant_id').drop_duplicates('variant_id')
            variants = variants[["variant_id", "target_type", "target_pref_name", "tid", "target_chembl_id",
                                 "protein_class_id", "accession", "component_id", "Classification",
                                 "UniProtID", "Status", "Protein names",
                                 "mutation", "Gene names", "Organism", "Length", "sequence"]]
            variants.rename(columns={'mutation': 'Mutation', 'sequence': 'Sequence'}, inplace=True)

            return variants

        process_and_write_chunks(variants,
                                 process_variants,
                                 f'{papyrus_version}_raw_variant_target_list_c{chembl_version}.txt',
                                 papyruslib_root, chembl_root)
        del variants

        # Bioactivities with target information
        bioactivities: PandasTextFileReader = pystow.load_df(papyruslib_root, chembl_root,
                                                             name=f'{papyrus_version}_raw_bioactivities_'
                                                                  f'only_small_molecules_conf_9_8_7_5_c{chembl_version}.txt',
                                                             read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})

        def merge_bioactivities_and_targets(bioactivities: pd.DataFrame, negate: bool = False):
            targets = pystow.load_df(papyruslib_root, chembl_root,
                                     name=f'{papyrus_version}_raw_target_list_c{chembl_version}.txt',
                                     read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})

            for chunk in targets:
                data = bioactivities.merge(chunk, on='tid')
                if not negate:
                    # Keep single proteins
                    yield data[data.target_type == 'SINGLE PROTEIN']
                else:
                    yield data[data.target_type != 'SINGLE PROTEIN']

        def merge_bioactivities_and_variant_targets(bioactivities: pd.DataFrame, negate: bool = False):
            variants = pystow.load_df(papyruslib_root, chembl_root,
                                      name=f'{papyrus_version}_raw_variant_target_list_c{chembl_version}.txt',
                                      read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
            # Process variants
            for chunk in variants:
                bioactivities_variants = (~bioactivities.variant_id.isna() &
                                          (bioactivities.variant_id != -1) &
                                          (bioactivities.variant_id != ''))
                if not bioactivities[bioactivities_variants].empty:
                    data = bioactivities[bioactivities_variants].merge(chunk, on='variant_id')
                    if not negate:
                        # Keep single proteins
                        yield data[data.target_type == 'SINGLE PROTEIN']
                    else:
                        yield data[data.target_type != 'SINGLE PROTEIN']
            del variants

            targets = pystow.load_df(papyruslib_root, chembl_root,
                                     name=f'{papyrus_version}_raw_target_list_c{chembl_version}.txt',
                                     read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
            # Process WT
            for chunk in targets:
                bioactivities_variants = (~bioactivities.variant_id.isna() &
                                          (bioactivities.variant_id != -1) &
                                          (bioactivities.variant_id != ''))
                if not bioactivities[~bioactivities_variants].empty:
                    data = bioactivities[~bioactivities_variants].merge(chunk, on='tid')
                    if not negate:
                        # Keep single proteins
                        yield data[data.target_type == 'SINGLE PROTEIN']
                    else:
                        yield data[data.target_type != 'SINGLE PROTEIN']
            del targets

        if include_variants:
            # Higher quality variant data
            process_and_write_chunks(bioactivities,
                                     merge_bioactivities_and_variant_targets,
                                     f'{papyrus_version}_raw_bioactivities_'
                                     f'only_small_molecules_conf_9_8_7_5_with_protein_accession_c{chembl_version}.txt',
                                     papyruslib_root, chembl_root)
            # Reinstantiate for lower quality variant data
            bioactivities = pystow.load_df(papyruslib_root, chembl_root,
                                           name=f'{papyrus_version}_raw_bioactivities_'
                                                f'only_small_molecules_conf_9_8_7_5_c{chembl_version}.txt',
                                           read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
            # Lower quality variant data
            process_and_write_chunks(bioactivities,
                                     partial(merge_bioactivities_and_targets, negate=True),
                                     f'{papyrus_version}_low_quality_raw_bioactivities_'
                                     f'only_small_molecules_conf_9_8_7_5_with_protein_accession_c{chembl_version}.txt',
                                     papyruslib_root, chembl_root)
        else:
            # Higher quality data
            process_and_write_chunks(bioactivities,
                                     merge_bioactivities_and_targets,
                                     f'{papyrus_version}_raw_bioactivities_'
                                     f'only_small_molecules_conf_9_8_7_5_with_protein_accession_c{chembl_version}.txt',
                                     papyruslib_root, chembl_root)
            # Reinstantiate for lower quality data
            bioactivities = pystow.load_df(papyruslib_root, chembl_root,
                                           name=f'{papyrus_version}_raw_bioactivities_'
                                                f'only_small_molecules_conf_9_8_7_5_c{chembl_version}.txt',
                                           read_csv_kwargs={'sep': '\t', 'chunksize': chunksize})
            # Lower quality data
            process_and_write_chunks(bioactivities,
                                     partial(merge_bioactivities_and_targets, negate=True),
                                     f'{papyrus_version}_low_quality_raw_bioactivities_'
                                     f'only_small_molecules_conf_9_8_7_5_with_protein_accession_c{chembl_version}.txt',
                                     papyruslib_root, chembl_root)

    # Molecules
    molstructures_query = """
    SELECT compound_structures.molregno,
           compound_structures.molfile
    FROM compound_structures
    """
    molecules: PandasTextFileReader = chembl_downloader.query(molstructures_query, version=chembl_version,
                                                              prefix=[papyruslib_root, chembl_root],
                                                              chunksize=chunksize)

    def process_molecules(molecules: pd.DataFrame):
        # Remove duplicates molregno
        molecules.drop_duplicates('molregno', inplace=True)
        # Parse CTAB and standardize molecules
        mols = molecules.molfile.apply(Chem.MolFromMolBlock)
        molecules.drop(columns=['molfile'])
        molecules['standardised_smiles'] = mols.apply(standardize)
        molecules['InChIKey'] = mols.apply(Chem.MolToInchiKey)
        molecules['InChI_AuxInfo'] = mols.apply(Chem.MolToInchiAndAuxInfo)[1]
        return molecules

    process_and_write_chunks(molecules,
                             process_molecules,
                             f'{papyrus_version}_raw_molecules_list_with_molreg_c{chembl_version}.txt',
                             papyruslib_root, chembl_root)
