import os.path

import numpy as np
import pandas as pd

from itertools import chain

import xgboost

from src.papyrus_scripts.reader import read_protein_set
from src.papyrus_scripts.modelling import qsar, pcm
from src.papyrus_scripts.preprocess import keep_quality, keep_source, keep_type, keep_protein_class, keep_accession, \
    keep_organism, consume_chunks

if __name__ == '__main__':
    coltypes = {'Activity_ID': object, 'Quality': object, 'source': object, 'CID': object,
                'SMILES': object, 'connectivity': object, 'InChIKey': object, 'InChI': object,
                'InChI_AuxInfo': object, 'target_id': object, 'accession': object, 'Protein_Type': object,
                'AID': object, 'type_IC50': object, 'type_EC50': object, 'type_KD': object,
                'type_Ki': object, 'type_other': object, 'Activity_class': object, 'relation': object,
                'pchembl_value': object, 'pchembl_value_Mean': np.float64, 'pchembl_value_StdDev': np.float64,
                'pchembl_value_SEM': np.float64, 'pchembl_value_N': np.float64, 'pchembl_value_Median': np.float64,
                'pchembl_value_MAD': np.float64}

    # data = pd.concat([chunk for chunk in tqdm(pd.read_csv("05.4_combined_set_without_stereochemistry.tsv.xz", sep='\t', chunksize=1000000, dtype=coltypes), total=60, ncols=80)], axis=0)
    # folder = 'C:/Users/brand/Documents/Uni/PhD/Papyrus/Dataset/'
    folder = 'F:/Downloads/Papyrus 05.4/'

    # descriptor_folder = 'C:/Users/brand/Documents/Uni/PhD/Papyrus/Dataset/'
    descriptor_folder = 'F:/Downloads/Papyrus 05.4/descriptors'

    # data = pd.read_csv(folder + "05.4_combined_set_without_stereochemistry.tsv.xz", sep='\t', nrows=10000)
    # results, models = qsar(data, descriptor_path=descriptor_folder)

    # Now results is a dataframe of the results
    # results.to_csv('results.txt', sep='\t')

    # free some memory
    # del data, results, models

    # Here we define chunksize of 100k
    # this requires that after applying filters, we call consume_chunks
    # chunksize     max allocated memory
    # 10 millions   6 Go
    #
    # data = pd.read_csv(folder + "05.4_combined_set_without_stereochemistry.tsv.xz", sep='\t', chunksize=50000, dtype=coltypes, low_memory=True)
    # Read protein annotations for filtering by classification
    # protein_data = read_protein_set(folder)

    # Apply filters (these are just stupid filters to debug)
    # filter1 = keep_quality(data, 'high')
    # filter2 = keep_source(filter1, 'ChEMBL29', True)
    # filter3 = keep_type(filter1, 'ic50')
    # filter4 = keep_protein_class(filter3, protein_data, [{'l2': 'Kinase'}, {'l5': 'Adenosine receptor'}])
    # filter4 = keep_protein_class(filter1, protein_data, {'l5': 'Adenosine receptor'})
    # filter5 = keep_accession(filter4, 'P00519')

    # Because data was chunked we need to call
    # consume_chunks to obtain the pandas dataframe
    # data_filtered = consume_chunks(filter4, total=1196)  # 1196 = 59,763,781 aggregated points / 50,000 chunksize

    # data_filtered.to_csv('results_filtered_adenosines_high.txt', sep='\t')


    # #QSAR
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results_filtered_adenosines_high.txt'), sep='\t',
                       index_col=0).sort_values('target_id')
    # results, models = qsar(data, descriptor_path=descriptor_folder, verbose=True,
    #                        model=xgboost.XGBClassifier(verbosity=0), stratify=True)
    # results.to_csv('QSAR_results_classification.txt', sep='\t')
    # results, models = qsar(data, descriptor_path=descriptor_folder, verbose=True,
    #                        model=xgboost.XGBRegressor(verbosity=0))
    # results.to_csv('QSAR_results_regression.txt', sep='\t')
    # # PCM
    result, model = pcm(data, mol_descriptor_path=descriptor_folder, prot_descriptor_path=descriptor_folder,
                        verbose=True, model=xgboost.XGBClassifier(verbosity=0), stratify=True)
    result.to_csv('PCM_results_classification.txt', sep='\t')
    # result, model = pcm(data, mol_descriptor_path=descriptor_folder, prot_descriptor_path=descriptor_folder,
    #                     verbose=True, model=xgboost.XGBRegressor(verbosity=0))
    # result.to_csv('PCM_results_regression.txt', sep='\t')

    # data = pd.read_csv(folder + "05.4_combined_set_without_stereochemistry.tsv.xz", sep='\t', chunksize=50000, dtype=coltypes, low_memory=True)
    # # Read protein annotations for filtering by classification
    # protein_data = read_protein_set(folder)
    #
    # # Apply filters (these are just stupid filters to debug)
    # filter1 = keep_quality(data, 'high')
    # filter2 = keep_protein_class(filter1, protein_data, [{'l2': 'Kinase'}])
    # filter3 = keep_organism(filter2, protein_data, 'Human', generic_regex=True)
    #
    #
    # # Because data was chunked we need to call
    # # consume_chunks to obtain the pandas dataframe
    # data_filtered = consume_chunks(filter3, total=1196)  # 1196 = 59,763,781 aggregated points / 50,000 chunksize

    # data_filtered.to_csv('results_filtered_kinases_human_high.txt', sep='\t')