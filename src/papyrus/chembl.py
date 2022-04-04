# -*- coding: utf-8 -*-

import os
from typing import Optional

import chembl_downloader
import pystow


def process_chembl_data(papyrus_version: str,
                        root_folder: Optional[str] = None,
                        chembl_version: Optional[int] = None) -> None:
    """Process data from ChEMBL to be integrated in Papyrus

    :param papyrus_version: The version of the Papyrus dataset being created.
    :param root_folder: Directory where intermediary files and ChEMBL SQLite will be stored
                        (default: pystow's home directory)
    :param chembl_version: Specific version of ChEMBL to be considered (default: latest)
    """
    # Determine default paths
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = root_folder
    papyruslib_root = pystow.module('papyrus-creation')
    # Execute the first query
    # chembl-downloader handles the download of the SQLite db if need be
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
    activity_df = chembl_downloader.query(activity_query, version=chembl_version, prefix='chembl')
    # Remove duplicates
    papyruslib_root.join(name=f'{papyrus_version}_raw_bioactivity_list_c{chembl_version}_potential_duplicates.txt')
    activity_df  = activity_df[activity_df.potential_duplicate != 1]
    # Remove invalid data
    #     data_validity_comment = 'Outside typical range'
    #     OR
    #     data_validity_comment = 'Non standard unit for type'
    #     OR
    #     data_validity_comment = 'Potential missing data'
    #     OR
    #     data_validity_comment = 'Potential transcription error'
    #     OR
    #     data_validity_comment = 'Potential author error'
    #     OR
    #     data_validity_comment = 'Author confirmed error';
    #     data_validity_comment != 'Author confirmed error';
    # >> $(version)_raw_bioactivity_list_c$(chembl_version)_questionned_valid_activity.txt
    # Keep measures with '=' sign
    #     standard_relation = '='
    # >> $(version)_raw_bioactivity_list_c$(chembl_version)_censored_activity.txt
    # pChEMBL value is defined
    #     strlength(pchembl_value) > 0;
    # >> $(version)_raw_bioactivity_list_c$(chembl_version)_undefined_pchembl_values.txt
    # Is activity questioned?
    #     activity_comment = 'inconclusive' OR
    #     activity_comment = 'unspecified' OR
    #     activity_comment = 'Indeterminate' OR
    #     activity_comment = 'Ineffective' OR
    #     activity_comment = 'Insoluble' OR
    #     activity_comment = 'Insufficient' OR
    #     activity_comment = 'Lack of solubility' OR
    #     activity_comment = 'inconclusive' OR
    #     activity_comment = 'Not Determined' OR
    #     activity_comment = 'ND(Insoluble)' OR
    #     activity_comment = 'tde' OR
    #     activity_comment = 'insoluble' OR
    #     activity_comment = 'not tested' OR
    #     activity_comment = 'uncertain' OR
    #     activity_comment = 'No compound available' OR
    #     activity_comment = 'No compound detectable' OR
    #     activity_comment = 'No data' OR
    #     activity_comment = 'Non valid test' OR
    #     activity_comment = 'Not assayed' OR
    #     activity_comment = 'OUTCOME = Not detected' OR
    #     activity_comment = 'Precipitate' OR
    #     activity_comment = 'Precipitated' OR
    #     activity_comment = 'Precipitates under the conditions of the study' OR
    #     activity_comment = 'Precipitation' OR
    #     activity_comment = 'Qualitative measurement' OR
    #     activity_comment = 'Too insoluble' OR
    #     activity_comment = 'Unable to be measured' OR
    #     activity_comment = 'Unable to calculate' OR
    #     activity_comment = 'Uncertain';
    # >> $(version)_raw_bioactivity_list_c$(chembl_version)_inconclusive_activity.txt
    # PubChem origin src_id = 7
    #   src_id = 7
    # >> $(version)_raw_bioactivity_list_c$(chembl_version)_pubchem_bioassays.txt
    # >> $(version)_raw_bioactivity_list_c$(chembl_version).txt
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
    # Rename chembl_id to compound_chembl_id, ...
    #   chembl_id >> compound_chembl_id
    #   pref_name >> compound_pref_name
    # Unique molregno
    # >> $(version)_raw_molecule_list_c$(chembl_version).txt

    # Read ChEMBL & PubChem bioactivities
    # Join molecules (above) on molregno
    # Is molecule type annotated
    #   molecule_type is defined
    # Is small molecule?
    #   molecule_type like "%small molecule%"
    # >> $(version)_raw_bioactivities_only_small_molecules_c$(chembl_version).txt
    # Read ChEMBL bioactivities
    #     Potential duplicates
    #     Questioned data validity
    #     Censored data
    #     Questioned activity
    #     Undefined activity
    # Join molecules (above) on molregno
    # Is molecule type annotated
    #   molecule_type is defined
    # Is small molecule?
    #   molecule_type like "%small molecule%"
    # >> $(version)_low_quality_raw_bioactivities_only_small_molecules_c$(chembl_version).txt
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
    # Separate AID, doc_id and year
    #     if year is defined and pubmed_id is defined => AID := chembl_id; doc_id := "PMID:" . pubmed_id;
    #     if year is defined and pubmed_id is not defined and doi is defined => AID := chembl_id; doc_id := "DOI:" . doi;
    #     if year is defined and pubmed_id is not defined and doi is not defined and patent is defined => AID := chembl_id; doc_id := "PATENT:" . RSubst(patent_id, '[^a-zA-Z0-9]', '', 'g');
    #     if year is defined and pubmed_id is not defined and doi is not defined and patent is not defined => AID := chembl_id; doc_id := chembl_id_1;
    #     if year is not defined and chembl_id_1 = "CHEMBL1201862" => Obtain data from "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/" . src_assay_id . "/dates/XML?dates_type=deposition";
    #                                                                 Parse XML into a table
    #                                                                   xroot = ET.fromstring(AllData.iloc[0, 0])
    #
    #                                                                   rows = []
    #                                                                   for info_node in xroot:
    #                                                                       aid = info_node[0].text
    #                                                                       year = info_node[1][0].text
    #                                                                       rows.append({"src_assay_id": aid, "Year": year})
    #
    #                                                                   AllData = pd.DataFrame(rows)
    #                                                                 Join on original data with src_assay_id
    #                                                                 AID: = chembl_id; doc_id: = "PubChemAID:".src_assay_id;
    #     elif year is not defined and chembl_id_1 = "CHEMBL1909046" => year := ""; AID := chembl_id; doc_id := "DrugMatrix:" . chembl_id_1;
    #     elif year is not defined and patent is defined => Obtain patent from Google big query
    #                                                       AID := chembl_id; doc_id := "PATENT:" . RSubst(patent_id, '[^a-zA-Z0-9]', '', 'g');
    #     elif RMatch(title, '\(\d{4}\)\s+PMID:') = True => #values := RMatch(title, '\((\d{4})\)\s+PMID:\s*(\d+)'); doc_id := "PMID:" . #values[3]; year := #values[2]; AID := chembl_id;
    #     else => AID := chembl_id; year := ""; doc_id := chembl_id_1;
    # Unique occurence od assay_id
    # >> $(version)_raw_assay_list_c$(chembl_version).txt



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