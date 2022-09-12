# -*- coding: utf-8 -*-

import os
import re
from typing import Optional

import pystow
import chembl_downloader
import pandas as pd

from .utils.pubchem import map_pubchem_assays
from .utils.patents import map_patent_id


BASE_DIR = os.path.dirname(__file__)

def process_chembl_data(papyrus_version: str,
                        chembl_version: Optional[int] = None,
                        prefix: Optional[str] = None) -> None:
    """Process data from ChEMBL to be integrated in Papyrus

    :param papyrus_version: The version of the Papyrus dataset being created.
    :param chembl_version: Specific version of ChEMBL to be considered (default: latest)
    :param prefix: Prefix directory, in pystow's data folder, where intermediary files
                   and ChEMBL SQLite will be stored.
    """
    # Determine default paths
    papyruslib_root = 'papyrus-creation'
    chembl_root = 'chembl'
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
    activity_df = chembl_downloader.query(activity_query, version=chembl_version,
                                          prefix=[papyruslib_root, chembl_root])
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
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_questionned_valid_activity.txt',
                   obj=questionned[submask])
    activity_df = activity_df[~mask]
    del questionned, submask
    # Keep measures with '=' sign
    mask = activity_df.standard_relation != '='
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_censored_activity.txt',
                   obj=activity_df[mask])
    activity_df = activity_df[~mask]
    # pChEMBL value is defined
    mask = activity_df.pchemb_value.isna()
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_undefined_pchembl_values.txt',
                   obj=activity_df[mask])
    activity_df = activity_df[~mask]
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
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_inconclusive_activity.txt',
                   obj=activity_df[mask])
    activity_df = activity_df[~mask]
    # PubChem origin src_id = 7
    mask = activity_df.src_id == 7
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_pubchem_bioassays.txt',
                   obj=activity_df[mask])
    activity_df = activity_df[~mask]
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}.txt',
                   obj=activity_df)
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
                                          prefix=[papyruslib_root, chembl_root])
    # Rename chembl_id to compound_chembl_id, ...
    compounds_df.rename(columns={'chembl_id': 'compound_chembl_id',
                                 'pref_name': 'compound_pref_name'},
                        inplace=True)
    # Unique molregno
    compounds_df.drop_duplicates(subset='molregno', inplace=True)
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_molecule_list_c{chembl_version}.txt',
                   obj=compounds_df)
    del compounds_df

    # Merge ChEMBL & PubChem bioactivities onto molecules
    activity_df = pd.concat([pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}.txt',
                                            read_csv_kwargs={'sep': '\t'}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_pubchem_bioassays.txt',
                                            read_csv_kwargs={'sep': '\t'})
                             ])
    compounds_df = pystow.load_df(papyruslib_root, chembl_root,
                                  name=f'{papyrus_version}_raw_molecule_list_c{chembl_version}.txt',
                                  read_csv_kwargs={'sep': '\t'})
    activity_df = activity_df.merge(compounds_df, on='molregno')
    # Keep small molecules
    activity_df = activity_df[activity_df.molecule_type.str.contains('small molecule')]
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_bioactivities_only_small_molecules_c{chembl_version}.txt',
                   obj=activity_df)
    del activity_df

    # Merge questionned bioactivities onto molecules
    activity_df = pd.concat([pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_potential_duplicates.txt',
                                            read_csv_kwargs={'sep': '\t'}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_questionned_valid_activity.txt',
                                            read_csv_kwargs={'sep': '\t'}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_censored_activity.txt',
                                            read_csv_kwargs={'sep': '\t'}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_undefined_pchembl_values.txt',
                                            read_csv_kwargs={'sep': '\t'}),
                             pystow.load_df(papyruslib_root, chembl_root,
                                            name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_inconclusive_activity.txt',
                                            read_csv_kwargs={'sep': '\t'})
                             ])
    activity_df = activity_df.merge(compounds_df, on='molregno')
    # Keep small molecules
    activity_df = activity_df[activity_df.molecule_type.str.contains('small molecule')]
    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_low_quality_raw_bioactivities_only_small_molecules_c{chembl_version}.txt',
                   obj=activity_df)
    del activity_df, compounds_df

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
    assay_df = chembl_downloader.query(assay_query, version=chembl_version,
                                       prefix=[papyruslib_root, chembl_root])
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
    data = data[['assay_id', 'description', 'assay_type', 'assay_test_type', 'relationship_type', 'tid', 'confidence_score', 'chembl_id', 'src_assay_id', 'AID', 'doc_id', 'year', 'variant_id']]
    data.rename(columns={'chembl_id': 'assay_chembl_id', 'year': 'Year'}, inplace=True)
    data.drop_duplicates('assay_id', inplace=True)

    pystow.dump_df(papyruslib_root, chembl_root,
                   name=f'{papyrus_version}_raw_assay_list_c{chembl_version}.txt',
                   obj=data)

    if not assay_df.empty:
        raise RuntimeError(f'Some assays were not considered by the filters:\n{assay_df}')

    # Read $(version)_raw_bioactivities_only_small_molecules_c$(chembl_version).txt
    # Remove doc_id
    # Join assays on assay_id
    # Confidence score assigned
    # confidence_score = '9' or confidence_score = '8' or confidence_score = '7' or confidence_score = '5'
    # >> $(version)_raw_bioactivities_only_small_molecules_conf_9_8_7_5_c$(chembl_version).txt
    # >> $(version)_low_quality_raw_bioactivities_only_small_molecules_conf_9_8_7_5_c$(chembl_version).txt

    # Read $(version)_low_quality_raw_bioactivities_only_small_molecules_c$(chembl_version).txt
    # Remove doc_id
    # Join assays on assay_id
    # >> append to $(version)_low_quality_raw_bioactivities_only_small_molecules_conf_9_8_7_5_c$(chembl_version).txt
    target_query = """
    SELECT target_dictionary.target_type,
           target_dictionary.pref_name,
           target_dictionary.tid,
           target_dictionary.chembl_id
    FROM target_dictionary
    """
    variant_mapping_query = """
    SELECT variant_sequences.mutation,
           variant_sequences.accession,
           variant_sequences.sequence
    FROM variant_sequences
    WHERE variant_sequences.variant_id = ?
    """
    target_mapping_query = """
    SELECT target_dictionary.target_type,
           target_dictionary.pref_name,
           target_dictionary.tid,
           target_dictionary.chembl_id
    FROM target_dictionary
    WHERE target_dictionary.tid = ?
    """
    molstructures_query = """
    SELECT compound_structures.molregno,
           compound_structures.molfile
    FROM compound_structures
"""