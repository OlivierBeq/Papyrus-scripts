# -*- coding: utf-8 -*-

import json
import os
import shutil
import sys
import time
import urllib
import gzip

import dask.dataframe as ddf
import dask.multiprocessing
from dask.delayed import delayed
import pandas as pd
import requests
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from rdkit import Chem, RDConfig
from rdkit.Chem import PandasTools

from .utils import IO, UniprotMatch

class Create_Environment(object):
    """
        Creates the workdirectory environment.
    """
    def __init__(self,data):
        if data['wd'] == None:
            return 

        if not os.path.exists(data['wd']):
            os.mkdir(data['wd'])
            
        else:
            if data['overwrite'] == True:
                shutil.rmtree(data['wd'])
                os.mkdir(data['wd'])

            else:
                print("Directory already exists, set overwrite to True to continue")
                print("Exiting now")

                sys.exit()

        # temporarily unpack .gz
        fn_in = data['p_in'] + '05.4_combined_set_without_stereochemistry.tsv.gz'
        fn_out = data['wd'] + '05.4_combined_set_without_stereochemistry.tsv'
        with gzip.open(fn_in, 'rb') as f_in:
            with open(fn_out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

class RCSB_data(object):
    def __init__(self):
        self.data = {
                            'pdb2InChI' : {},
                            'InChI2pdb' : {},
                            'cc2pdb'    : {},
                            'pdb2cc'    : {}
                        }

class Papyrus(object):
    def __init__(self, environment):
        self.environment = environment
        self.read_RCSB_json()
        self.read_papyrus_data()

    def read_RCSB_json(self):
        if self.environment['js_in'] != None:
            self.RCSB_data = IO.read_jsonfile(self.environment['js_in'])                    

        else:
            self.RCSB_data = IO.read_jsonfile(self.environment['js_out']) 

    def read_papyrus_data(self):
        self.rcsb_data_list = []
        self.InChI2PDB()
        print("reading data from Papyrus sdf")
        papyrus_tsv = self.environment['wd'] + '05.4_combined_set_without_stereochemistry.tsv'

        self.ddf = ddf.read_csv(papyrus_tsv,  # read in parallel
                                sep='\t', 
                                #blocksize=64000000,
                                dtype={'Activity_class': 'object',
                                       'pchembl_value_N': 'float64',
                                       'Year' : 'float64',
                                       'type_EC50': 'object',
                                       'type_KD' : 'object',
                                       'type_Ki' : 'object' 
                                       }
                               )

        with ProgressBar():
            df_rcsb = self.ddf.map_partitions(self.match2rcsb).compute()
            df_rcsb.dropna(subset=['RCSB_accesion_code'], inplace=True)

        # Now do the matching ==> this needs to be a list 
        pdbs = self.RCSB_data['pdb2cc'].keys()
        pdbs = [pdb for line in pdbs for pdb in line.split()]

        pdb_df = UniprotMatch.uniprot_mappings(pdbs,map_from='PDB_ID',map_to='ACC')

        # tmp csv writing
        pdb_df.to_csv(self.environment['wd'] + 'tmp_RCSB.csv',sep='\t',index=False)

        # load it
        pdb_df = pd.read_csv(self.environment['wd'] + 'tmp_RCSB.csv', sep = '\t')

        # construct pdb:uniprot look up
        pdb_dict = {}
        for index, row in pdb_df.iterrows():
            if not row['PDB_ID'] in pdb_dict:
                pdb_dict[row['PDB_ID']] = [row['ACC']]

            else:
                pdb_dict[row['PDB_ID']].append(row['ACC'])

        #to_pop = []

        tmp = []
        for index, row in df_rcsb.iterrows():
            pdb_tmp = []

            for pdb in row['RCSB_accesion_code'].split(';'):
                if pdb in pdb_dict:
                    if row['Activity_ID'].split('_')[2] in pdb_dict[pdb]:
                        pdb_tmp.append(pdb)

            # concatenate pdb list
            if len(pdb_tmp) > 0:
                pdb_str = ';'.join(pdb_tmp)

            
            else:
                pdb_str = None

            tmp.append(pdb_str)
                
        df_rcsb['RCSB_matched'] = tmp

        df_rcsb.dropna(subset=['RCSB_matched'], inplace=True)
        df_rcsb.drop(['RCSB_accesion_code'],axis=1, inplace=True)
        df_rcsb.rename({'RCSB_matched':'RSCB_accesion_code'},inplace=True)
        df_rcsb.to_csv(self.environment['wd'] + self.environment['p_out'], sep='\t',index=False)

    def InChI2PDB(self):
        tmp = {}
        for InChI in self.RCSB_data['InChI2pdb']:
            lig = self.RCSB_data['InChI2pdb'][InChI]
            if lig in self.RCSB_data['cc2pdb']:
                tmp[InChI] = self.RCSB_data['cc2pdb'][lig]

        self.RCSB_data['InChI2PDB'] = tmp

    def match2rcsb(self,df):
        df_RCSB = df.loc[df['InChI'].isin(self.RCSB_data['InChI2pdb'].keys())]
        df_RCSB['RCSB_accesion_code'] = df_RCSB['InChI'].map(self.RCSB_data['InChI2PDB'])

        return(df_RCSB)

class RCSB(object):
    def __init__(self, data):
        """ Object to interact with RCSB webserver
            
        """ 
        self.environment = data
        self.base_url = 'http://ligand-expo.rcsb.org/dictionaries/{}'
        self.RCSB_data = RCSB_data()

        if self.environment['js_in'] == None:
            self.get_InChI()
            self.get_PDBID()

        else:
            # Check if the json file provided actually is OK:
            self.read_json()

        self.write_json()

    def read_json(self):
        try:
            jsonfile = self.environment['js_in']

        except:
            print("Couldn't read json file, please provide a valid file")
            sys.exit()

    def write_json(self):
        # TO DO, add working directory
        jsonfile = self.environment['js_out']
        IO.write_jsonfile(self.RCSB_data.data, jsonfile) 

    def run_query(self,query):
        url = self.base_url.format(query)
        response = urllib.request.urlopen(url).read()

        # Responses are flat text
        data = response.splitlines()

        return data

    def get_InChI(self):
        """
            Get PDB ligand codes from InChI
        
        """
        data = self.run_query('Components-inchi.ich')

        for line in data:
            line = line.decode('utf-8')
            if not 'InChI' in line:
                continue
            line = line.split("\t")
            self.RCSB_data.data['pdb2InChI'][line[1]] = line[0]
            self.RCSB_data.data['InChI2pdb'][line[0]] = line[1]

    def get_PDBID(self):
        """
            Needs a .json query for rcsb (https://search.rcsb.org/#search-api)
        
        """
        data = self.run_query('cc-to-pdb.tdd')

        for line in data:
            line = line.decode('utf-8')
            line = line.split("\t")
            self.RCSB_data.data['pdb2cc'][line[1]] = line[0]
            # Change delimiter
            pdb = line[1].split()
            pdb = ';'.join(pdb)
            self.RCSB_data.data['cc2pdb'][line[0]] = pdb

class Cleanup(object):
    def __init__(self, data):
        """ Cleanup working files
            
        """ 
        self.environment = data

        os.remove(self.environment['wd'] + 'tmp_RCSB.csv')
        os.remove(self.environment['wd'] + '05.4_combined_set_without_stereochemistry.tsv')

class Init(object):
    def __init__(self, data):
        """ Retrieves a dictionary of user input from matchRCSB.py:
               {
                 'wd'       : wd,
                 'js_in'    : js_in,
                 'js_out'   : js_out,
                 'p_in'     : p_in,
                 'p_out'    : p_out,
               }
        """ 
        # Globals + command line stuff  
        self.environment = data
        Create_Environment(self.environment)
        RCSB(self.environment)
        Papyrus(self.environment)
        Cleanup(self.environment)
