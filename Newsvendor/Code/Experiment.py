"""

This module contains classes for running experiments using various models from the dddex package.

"""

    

# Imports
import numpy as np
import pandas as pd
from scipy import stats
import copy
import time
import datetime as dt
import os
import pickle
import json
import joblib
from joblib import dump, load

import itertools
import contextlib
from tqdm import tqdm

import gurobipy as gp
from gurobipy import GRB


import wandb


# Import dddex

# ...




class PreProcessing:
    

    """
    ...
    
    """

    # Initialize
    def __init__(self, **kwargs):

        """
        ...
        
        """

        # Default experiment setup
        self.experiment_setup = dict()


class Experiment:

    """
    ...
    
    """

    # Initialize
    def __init__(self, **kwargs):

        """
        ...
        
        """

        # Default experiment setup
        self.experiment_setup = dict()
        
        
        
        
        
class PostProcessing:  
    
    """
    ...
    
    """

    # Initialize
    def __init__(self, **kwargs):

        """
        ...
        
        """

        # Default experiment setup
        self.experiment_setup = dict()
        
        
    def get_from_wandb(self, 
                       api:wandb.Api(), 
                       entity:list, 
                       project:list, 
                       artifact:list = ['costData', 'decisionData'],
                       alias:list = ['latest'], 
                       results_dir:str = '.', 
                       project_prefix:list = [''], 
                       **kwargs):
        
        """
        
        This function uses an API connection to weights & biases and downloads project results. The function 
        will look up artifcats on weights & biases stored under:
        
        <entity>/<project_prefix><project>/<artifact>:<alias>
        
        and store the files under <results_dir>/<artifact>/
   
        
        Arguments:
        
            api: logged-in API connection to weights & biases
            entity: entity name as used on weights & biases
            project: project name for which to download data
            artifact: project name for which to download data
            alias: alias of the data to be downloaded
            results_dir: local directory where to store results
            project_prefix: specifying a project prefix
            kwargs: ignored
        
        Returns:
        
            None
        
        """

        
        # Iterate over full permutation of all arguments
        for entity_ in entity:
            for project_prefix_ in project_prefix:
                for project_ in project:
                    for artifact_ in artifact:
                        for alias_ in alias:
                            
                            # Create paths
                            wandb_path = entity_+'/'+project_prefix_+project_+'/'+artifact_+':'+alias_
                            results_path = results_dir+'/'+project_+'/'+artifact_
                            
                            # Get and download
                            data_connection = api.artifact(wandb_path)
                            data_connection.download(results_path)
                            
        return None

        
        
    def aggregate(self, path, remove_files=False, save_results=True, **kwargs):
        
        """
        
        This function looks up all .pkl files in the folder under 'path' and
        aggregates the results in these files into one pandas data frame.
        
        The .pkl have a given, expected structure of (id, decisionType, costs).
        
        Arguments:
        
            path(str): string specifying the path of the folder where .pkl files are stored
            remove_files(bool)=False: should all source files be deleted after aggregating?
            save_results(bool)=True: should the result file be saved as .csv in folder under path?
            kwargs(dict): can provide name to save the file using name='...'
        
        Returns:
        
            results(pd.DataFrame) of aggregated results from all .pkl files in path
        
        """
    
        # Get all files and combine to one data set
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        results = pd.DataFrame()

        for file in files:

            if file.endswith(".pkl"):

                with open(path+'/'+file, 'rb') as f:
                    results_ = pickle.load(f).reset_index()

                results_['model'] = file.replace('.pkl', '')

                results = pd.concat([results, results_])

        # Extract service level
        results['sl'] = results['decisionType'].astype('string').apply(
            lambda s: int(s.replace('quantile_', '')) / 1000
        )

        # Extract coefficient of prescriptiveness
        results['coPres'] = results['costs']

        # Remove all files
        if(remove_files):

            for file in files:

                if file.endswith(".pkl"):

                    os.remove(path+'/'+file)

        # Save
        if(save_results):
            
            name = kwargs['name']+'_' if 'name' in kwargs else ''

            results.to_csv(path+'/'+name+'results.csv', sep=',', index=False)

        return results



        
        
        
        
        
class Evaluation:
    
    """
    ...
    
    """

    # Initialize
    def __init__(self, **kwargs):

        """
        ...
        
        """

        # Default experiment setup
        self.experiment_setup = dict()
        
        
      
    
        