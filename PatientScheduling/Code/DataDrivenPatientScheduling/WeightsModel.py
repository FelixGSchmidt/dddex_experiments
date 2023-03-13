"""

...


"""

# Import utils
import numpy as np
import pandas as pd
import time
import copy
import datetime as dt
import joblib
import contextlib
from tqdm import tqdm

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from matplotlib.patches import PathPatch

# sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# dddex
from dddex.crossValidation import QuantileCrossValidation
from dddex.levelSetKDEx_univariate import LevelSetKDEx, LevelSetKDEx_NN
from dddex.wSAA import RandomForestWSAA, SampleAverageApproximation

# Import Gurobi
import gurobipy as gp
from gurobipy import GRB




    

class WeightsModel:
    
    """
    
    ...
    
    """
        
    ### Init
    def __init__(self, **kwargs):

        """
 

            
        """

        # Set params
        self.params = {
                    
        }
        
        

    ### Function to set params
    def set_params(self, **kwargs):
        
        # Update all items that match an existing key
        self.params.update((p, kwargs[p]) for p in set(kwargs).intersection(self.params))
            
            
        
    ### Function to get params
    def get_params(self):
        
        return self.params
    
    
    
    def tune_point_estimator(self, X_train, y_train, estimator, cv_folds, hyper_params_grid, 
                             tuning_params, random_search=True, print_time = False, **kwargs):
    
        # Timer
        start_time = dt.datetime.now().replace(microsecond=0)
        st_exec = time.time()
        st_cpu = time.process_time() 

        # Tuning approach
        if random_search:

            # Random search CV
            cv_search = RandomizedSearchCV(estimator=estimator,
                                           cv=cv_folds,
                                           param_distributions=hyper_params_grid,
                                           **tuning_params)

        else:

            # Grid search SV
            cv_search = GridSearchCV(estimator=estimator,
                                     cv=cv_folds,
                                     param_grid=hyper_params_grid,
                                     **tuning_params)

        # Fit the cv search
        cv_search.fit(X_train, y_train) 

        # CV results
        best_estimator = cv_search.best_estimator_
        
        # Timer
        exec_time_sec = time.time()-st_exec
        cpu_time_sec = time.process_time()-st_cpu
        
        # Status
        if print_time: 
            print('Time:', dt.datetime.now().replace(microsecond=0) - start_time)  
            print('>> Execution time:', np.around(exec_time_sec, 0), "seconds") 
            print('>> CPU time:', np.around(cpu_time_sec, 0), "seconds") 


        return best_estimator
    
    
    
    
    
    
    def tune_density_estimator(self, X_train, y_train, estimator, cv_folds, hyper_params_grid, 
                               tuning_params, random_search = False, print_time = False, **kwargs):

        # Timer
        start_time = dt.datetime.now().replace(microsecond=0)
        st_exec = time.time()
        st_cpu = time.process_time() 

        # Tuning approach
        cv_search = QuantileCrossValidation(estimator = estimator, 
                                            cvFolds = cv_folds,
                                            parameterGrid = hyper_params_grid,
                                            randomSearch = random_search,
                                            **tuning_params)

        # Fit the cv search
        cv_search.fit(X_train, y_train)

        # CV results
        best_estimator = cv_search.bestEstimator
        
        # Timer
        exec_time_sec = time.time()-st_exec
        cpu_time_sec = time.process_time()-st_cpu
        
        # Status
        if print_time: 
            print('Time:', dt.datetime.now().replace(microsecond=0) - start_time)  
            print('>> Execution time:', np.around(exec_time_sec, 0), "seconds") 
            print('>> CPU time:', np.around(cpu_time_sec, 0), "seconds") 

        return best_estimator




    