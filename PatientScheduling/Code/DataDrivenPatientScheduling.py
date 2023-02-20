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

# sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# dddex
from dddex.crossValidation import QuantileCrossValidation

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




    
    

#### Patient Scheduling
class PatientScheduling:
    
    """
    
    This class provides the data-driven optimization module for patient scheduling with fixed sequence. It uses sample
    average approximation (SAA) over scenarios.
    
    """
        
    ### Init
    def __init__(self, c_waitingtime=1, c_overtime=2, T=None, LogToConsole=0, Threads=1, **kwargs):

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
    
    
    
    
    

#### ...
class Experiment:

    """
    ...
    
    """

    ## Initialize
    def __init__(self, **kwargs):

        """
        ...
        
        """
        
        return None



    #### Function to load experiment data
    def load_data(self, **kwargs):


        return None

    
    
    
    #### Function to pre-process experiment data
    def preprocess_data(self, **kwargs):

        """

        ...


        """

        return None





    #### Function to run data-driven patient scheduling experiment
    def run(self, X, y, date, dates, areas, weightsModel = None, cost_params = {'CR': 0.50, 'c_waiting_time': 1, 'c_overtime': 1}, 
            gurobi_params = {'LogToConsole': 0, 'Threads': 32}, K = 10**3, alpha = 1, print_status = False):

        
    
        """

        ...

        Arguments:
    
            X: ...
            y: ...
            date: ...
            dates: ...
            areas: ...
            weightsModel = None: ...
            cost_params = {'CR': 0.50, 'c_waiting_time': 1, 'c_overtime': 1}: ...
            gurobi_params = {'LogToConsole': 0, 'Threads': 32}: ...
            K = 10*3: ...
            alpha = 1: ...
            print_status = True: ...

        Returns:

            results(pd.DataFrame): ...


        """

        # Train-test split
        y_train, y_test = y[dates < date].flatten(), y[dates == date].flatten()
        X_train, X_test = X[dates < date], X[dates == date]
        dates_train, dates_test = dates[dates < date], dates[dates == date]
        areas_train, areas_test = areas[dates < date], areas[dates == date]

        # Get time budget per area
        hist_durations = pd.DataFrame({
            'duration': y_train, 
            'area': areas_train,
        }).groupby('area').agg(
            median_duration=('duration', np.median)
        ).reset_index()
        
        hist_durations = dict(zip(hist_durations.area, hist_durations.median_duration))
        
        # LSx and wSAA
        if not str(type(weightsModel)) == "<class 'dddex.wSAA.SampleAverageApproximation'>":

            # Fit weights model
            weightsModel.fit(X_train, y_train)


        # Initialize
        results = pd.DataFrame()

        # For each area 
        for area in list(set(areas_test)):

            # Initialize
            results_ = {}

            # Progress
            if print_status:
                print('=====================================================================================================')
                print('# Area:',area)

            # Timer
            weights_st_exec = time.time()
            weights_st_cpu = time.process_time() 

            # Select test data for current area
            y_test_ = y_test[area == areas_test]
            X_test_ = X_test[area == areas_test]

            # Number of patient cases to schedule
            M = len(y_test_)

            # Set random sequence of patient cases
            #patient_sequence = np.arange(M)
            #np.random.shuffle(patient_sequence)

            #y_test_ = y_test_[patient_sequence]
            #X_test_ = X_test_[patient_sequence]

            # Set time budget (based on median duration per area x number of cases on test day)
            T = hist_durations[area] * M * alpha
            
            # SAA
            if str(type(weightsModel)) == "<class 'dddex.wSAA.SampleAverageApproximation'>":
                
                # Fit SAA on data of current area
                weightsModel.fit(y_train[area == areas_train])
                
            # Get estimated conditional distribution
            conditionalDistribution = weightsModel.getWeights(X_test_, outputType='summarized')

            # Timer
            weights_exec_time_sec = time.time()-weights_st_exec
            weights_cpu_time_sec = time.process_time()-weights_st_cpu

            # Generate K scenarios
            scenarios = []

            # Timer
            scenarios_st_exec = time.time()
            scenarios_st_cpu = time.process_time() 

            # Draw
            for k in range(K):    

                # For each patient
                scenario = []
                for j in range(M):

                    # Weighted samples
                    weights = conditionalDistribution[j][0]
                    samples = conditionalDistribution[j][1]

                    # Add scenario for patient j
                    scenario += [np.random.choice(samples.flatten(), p=weights.flatten())]

                # Add scenarios
                scenarios += [scenario]

            # Scenarios
            scenarios = np.array(scenarios)   

            # Timer
            scenarios_exec_time_sec = time.time()-scenarios_st_exec
            scenarios_cpu_time_sec = time.process_time()-scenarios_st_cpu

            # for each cost param setting
            for cost_params_ in cost_params:

                # Progress
                if print_status:
                    print('## Cost param setting:',cost_params_)

                # Timer
                opt_st_exec = time.time()
                opt_st_cpu = time.process_time() 

                # Optimization
                CR, c_waiting_time, c_overtime = cost_params_['CR'], cost_params_['c_waiting_time'], cost_params_['c_overtime']

                # Initialize optimization model
                ddps = PatientScheduling(c_waitingtime=c_waiting_time, c_overtime=c_overtime, T=T, **gurobi_params)

                # Set up optimization model with scenarios
                ddps.create(z=scenarios)

                # Solve
                schedule, times, status, solutions, gap = ddps.optimize()

                # Waiting time
                scheduled_start, scheduled_duration, actual_duration, waiting_time = [], [], [], []    

                for j in range(M):

                    scheduled_start += [schedule[j]]
                    scheduled_duration += [times[j]]
                    actual_duration += list(y[j])
                    waiting_time += [0] if j == 0 else [max([0]+[waiting_time[j-1]+actual_duration[j-1]-scheduled_duration[j-1]])]
         
                # Overtime (using last j)
                overtime = max([0]+[waiting_time[j]+actual_duration[j]-scheduled_duration[j]])

                # Cost
                cost = c_waiting_time * sum(waiting_time) + c_overtime * overtime

                # Timer
                optimization_exec_time_sec = time.time()-opt_st_exec
                optimization_cpu_time_sec = time.process_time()-opt_st_cpu

                # Store results
                result = {

                    'date': date,
                    'area': area,
                    'n_patients': M,
                    'n_scenarios': K,
                    'CR': CR,
                    'c_waiting_time': c_waiting_time,
                    'c_overtime': c_overtime,
                    'historical_time_budget': hist_durations[area] * M,
                    'time_budget_multiplier': alpha,
                    'total_time_budget': T,
                    'waiting_time': sum(waiting_time),
                    'overtime': overtime,
                    'cost': cost,
                    'scheduled_total_durations': sum(scheduled_duration),
                    'actual_total_durations': sum(actual_duration),
                    'optimization_exec_time_sec': optimization_exec_time_sec,
                    'optimization_cpu_time_sec': optimization_cpu_time_sec,
                    'scenarios_exec_time_sec': scenarios_exec_time_sec,
                    'scenarios_cpu_time_sec': scenarios_cpu_time_sec,
                    'weights_exec_time_sec': weights_exec_time_sec,
                    'weights_cpu_time_sec': weights_cpu_time_sec

                }

                # Add to results
                results_[CR] = copy.deepcopy(result)   

            # Create data frame results
            results = pd.concat([results, pd.DataFrame.from_dict(results_, orient='index').reset_index(drop=True)])    

        # Progress
        if print_status:
            print('=====================================================================================================')
            print('>>>> Done in', dt.datetime.now().replace(microsecond=0) - start_time)

        return results



    
    
    
    
    


   

    
    @contextlib.contextmanager
    def tqdm_joblib(self, tqdm_object):

        """

        Context manager (Credits: 'https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution')

        Arguments:

            ...

        Returns:

            ...

        """


        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()


