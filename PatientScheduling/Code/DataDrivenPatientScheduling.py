"""

...


"""


# Import utils
import numpy as np
import pandas as pd
import math
import time
import json
import pyreadr
import pickle
from joblib import dump, load
import os
import copy


# Import Gurobi
import gurobipy as gp
from gurobipy import GRB




    


    


#### Patient Scheduling
class PatientScheduling:
    
    """
    
    This class provides the data-driven optimization module for patient scheduling with fixed sequence. It uses sample
    average approximation (SAA) over scenarios.
    
    """
        
    ### Init
    def __init__(self, c_waitingtime=1, c_overtime=2, T=None, LogToConsole=1, Threads=1, **kwargs):

        """
        
        Initializes class.
        
        Arguments:
        
            c_waitingtime(int): penalty cost for waiting time per minute
            c_overtime(int): penalty cost for overtime per minute
            T(int): total time budget in minutes
            
            LogToConsole(int): Gurobi log
            Threads(int): max number of threads to use (max=32)
            
        Further key word arguments (kwargs): ignored

            
        """

        # Set params
        self.params = {
            
            'c_waitingtime': c_waitingtime,
            'c_overtime': c_overtime,
            'T': T,
            'LogToConsole': LogToConsole,
            'Threads': Threads
        
        }
        
        

    ### Function to set params
    def set_params(self, **kwargs):
        
        # Update all items that match an existing key
        self.params.update((p, kwargs[p]) for p in set(kwargs).intersection(self.params))
            
            
        
    ### Function to get params
    def get_params(self):
        
        return self.params
        
    
        
    ### Function to create and set up the model
    def create(self, z, **kwargs):

        """
        
        Initialize and set up patient scheduling optimization task. 
        
        Arguments:
            
            z(np.array(K, M): K scenarios for sequence of M patients

        Further key word arguments (kwargs): passed to set_params() to update (valid) paramaters
        

        """

        # Set params
        self.set_params(**kwargs)
     
        # Number of scenarios
        K = z.shape[0]
        
        # Number of patients
        M = z.shape[1]
        
        # Problem parameters
        c_waitingtime = self.params['c_waitingtime']
        c_overtime = self.params['c_overtime']
        T = self.params['T']

        
        
        ## Create model
        if self.params['LogToConsole']==0:
            
            env = gp.Env(empty=True)
            env.setParam('OutputFlag',0)
            env.start()
            self.m = gp.Model(env=env)
            
        else:
            
            self.m = gp.Model()
            
            
        ## Create decision variables
        
        # Primary decision variable (sequential time budget)
        x = self.m.addVars(M, vtype='I', lb=0, name='x')

        # Auxiary decision variable (waiting time)
        w = self.m.addVars(K, M, vtype='I', lb=0, name='w')        

        # Auxiary decision variable (overtime)
        l = self.m.addVars(K, vtype='I', lb=0, name='l') 


        
        
        ## Set objective function 
        OBJ = self.m.setObjective(

            # Sample average
            gp.quicksum(

                1/K * (                                         

                    # Waiting time
                    c_waitingtime * gp.quicksum(w[k,j] for j in range(1, M)) + 

                    # Overtime
                    c_overtime * l[k]


                ) for k in range(K)),        

            # min
            GRB.MINIMIZE
        )


        ## Set constraints
        
        # Waiting times
        C1 = self.m.addConstrs(

            w[k,j] + z[k,j] - x[j] <= w[k,j+1]

            for j in range(M-1)
            for k in range(K)

        )  

        # Overtime
        C2 = self.m.addConstrs(

            w[k,M-1] + z[k,M-1] + x[M-1] <= l[k]

            for k in range(K)

        )    

        # Total time budget
        if not T is None:
            
            C3 = self.m.addConstr(

                gp.quicksum(x[j] for j in range(M)) <= T

            )    

        
    
    ### Function dump model
    def dump(self):
        
        self.m = None

        
    ### Function to optimize model
    def optimize(self, **kwargs):
        
        """
        
        Finds optimal solution by minimzing sample average.
        
        
        Arguments: None
            
        Further key word arguments (kwargs): passed to update Gurobi meta params (other kwargs are ignored)

        Returns:
        
            ...
            
        """

        # Update gurobi meta params if provided        
        self.set_params(**{
            
            'LogToConsole': kwargs.get('LogToConsole', self.params['LogToConsole']),
            'Threads': kwargs.get('Threads', self.params['Threads'])
            
        })           
            
        # Set Gurobi meta params
        self.m.setParam('LogToConsole', self.params['LogToConsole'])
        self.m.setParam('Threads', self.params['Threads'])

        # Optimize
        self.m.optimize()     
        
        # Solution
        if self.m.SolCount > 0:
        
            times = [var.xn for var in self.m.getVars() if 'x' in var.VarName]

        else:
            
            times = np.nan
            
        # Schedule (start times created using cumulative time budgets, i.e., 0, x_{1}, x_{1} + x_{2}, ..., x_{1} + ... + x_{M-1})
        schedule = np.cumsum([0] + times[0:(len(times)-1)])
            
        
        # Solution meta data
        status = self.m.status
        solutions = self.m.SolCount
        gap = self.m.MIPGap
        
                    
        # return decisions
        return schedule, times, status, solutions, gap
    
    
    
    

    