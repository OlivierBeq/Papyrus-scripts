# -*- coding: utf-8 -*-

import json
import os
import shutil
import sys
import time
import urllib

import dask.dataframe as ddf
import dask.multiprocessing
import pandas as pd
import requests
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from rdkit import Chem, RDConfig
from rdkit.Chem import PandasTools

from src.papyrus_scripts.utils import IO, UniprotMatch


class Create_Environment(object):
    """
        Creates the workdirectory environment.
    """
    def __init__(self,wd, overwrite=False):
        if wd == None:
            return 

        if not os.path.exists(wd):
            os.mkdir(wd)
            
        else:
            if overwrite == True:
                shutil.rmtree(wd)
                os.mkdir(wd)

            else:
                print("Directory already exists, set overwrite to True to continue")
                print("Exiting now")

                sys.exit()

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
        # This will later take # CPUs from settings class
        #self.client = Client(n_workers=2)

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
        papyrus_tsv = self.environment['p_in']
        self.ddf = ddf.read_csv(papyrus_tsv,  # read in parallel
                                sep='\t', 
                                blocksize=64000000,
                                dtype={'Activity_class': 'object',
                                       'pchembl_value_N': 'float64'}
                               )
        with ProgressBar():
            df_rcsb = self.ddf.map_partitions(self.match2rcsb).compute()
            df_rcsb.dropna(subset=['RCSB_accesion_code'], inplace=True)
        # to be put in function
        pdbs = self.RCSB_data['pdb2cc'].keys()
        pdbs = [i.split(' ', 1)[0] for i in pdbs]

        #pdb_df = UniprotMatch.UniprotMatch(pdbs,mapfrom='PDB_ID',mapto='ACC')
        
        # tmp csv writing
        #pdb_df.to_csv('tmp_RCSB.csv',sep='\t',index=False)


        # load it
        pdb_df = pd.read_csv('tmp_RCSB.csv', sep = '\t')
        pdb_dict = dict(zip(pdb_df.PDB_ID, pdb_df.ACC))

        # How slow is it to actually loop?
        to_pop = []
        for index, row in df_rcsb.iterrows():
            pdb_tmp = []
            for pdb in row['RCSB_accesion_code'].split(';'):
                if pdb in pdb_dict:
                    if pdb_dict[pdb] == row['Activity_ID'].split('_')[2]:
                        pdb_tmp.append(pdb)
            
            if len(pdb_tmp) > 0:
                pdb_str = ';'.join(pdb_tmp)
                #row['RCSB_accesion_code'] = pdb_str
                df_rcsb.at[index,'RCSB_accesion_code']=pdb_str

            else:
                # delete the row
                to_pop.append(index)

        df_rcsb.drop(to_pop, inplace = True)
        df_rcsb.to_csv(self.environment['p_out'], sep='\t',index=False)

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
        Create_Environment(self.environment['wd'])
        RCSB(self.environment)
        Papyrus(self.environment)