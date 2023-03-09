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




    
    

#### Patient Scheduling
class PatientScheduling:
    
    """
    
    This class provides the data-driven optimization module for patient scheduling with fixed sequence. 
    It uses sample average approximation (SAA) over scenarios.
    
    """
        
    ### Init
    def __init__(self, c_waiting_time=1, c_overtime=1, T=None, LogToConsole=0, Threads=1, **kwargs):

        """
        
        Initializes class.
        
        Arguments:
        
            c_waiting_time(int): penalty cost for waiting time per minute
            c_overtime(int): penalty cost for overtime per minute
            T(int): total time budget in minutes
            
            LogToConsole(int): Gurobi log
            Threads(int): max number of threads to use (max=32)
            
        Further key word arguments (kwargs): ignored

            
        """

        # Set params
        self.params = {
            
            'c_waiting_time': c_waiting_time,
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
        c_waiting_time = self.params['c_waiting_time']
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
                    c_waiting_time * gp.quicksum(w[k,j] for j in range(1, M)) + 

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

            w[k,M-1] + z[k,M-1] - x[M-1] <= l[k]

            for k in range(K)

        )    

        # Total time budget
        if not T is None:
            
            C3 = self.m.addConstr(

                gp.quicksum(x[j] for j in range(M)) <= T

            )    

        # Waiting time of first patient is always zero
        C4 = self.m.addConstrs(

            w[k,0] == 0

            for k in range(K)
            
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
            
        # Schedule (start times created using cumulative time budgets)
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


    def get_time_budget(self, y_train, areas_train, rho, which = None):
    
        """


        ...

        Arguments:

            y_train: np.array(n,) durations in training data
            areas_train: np.array(n,) treatment area for each value in y_train
            rho: utilization (duration scaling factor)
            which = None: one of 'median' or 'mean' defining which calculation logic to apply

        """

        # Ensure we have flat np.arrays
        y_train = np.array(y_train).flatten()
        areas_train = np.array(areas_train).flatten()

        # Check inputs
        if len(y_train) != len(areas_train):

            raise ValueError('y_train and areas_train have to be of the same length.')


        ## Based on median historical duration per area

        # Historical durations
        hist_durations = pd.DataFrame({
            'duration': y_train, 
            'area': areas_train,
        }).groupby('area').agg(
            median_duration=('duration', np.median)
        ).reset_index()

        median_durations_by_area = dict(zip(hist_durations.area, hist_durations.median_duration / rho))


        ## Based on mean historical duration per area

        # Historical durations
        hist_durations = pd.DataFrame({
            'duration': y_train, 
            'area': areas_train,
        }).groupby('area').agg(
            mean_duration=('duration', np.mean)
        ).reset_index()

        mean_durations_by_area = dict(zip(hist_durations.area, hist_durations.mean_duration / rho))


        ## Finalize
        time_budgets = {'median': median_durations_by_area, 'mean': mean_durations_by_area}

        return time_budgets.get(which, time_budgets)


    
    
    
    
    #### Function to run data-driven patient scheduling experiment
    def run(self, X, y, date, dates, areas, weightsModel = None, 
            cost_params = {'c_waiting_time': 1, 'c_overtime': 1}, 
            gurobi_params = {'LogToConsole': 0, 'Threads': 32}, 
            K = 10**3, rho = 1, print_status = False):

        
    
        """

        ...

        Arguments:
    
            X: ...
            y: ...
            date: ...
            dates: ...
            areas: ...
            weightsModel = None: ...
            cost_params = {'c_waiting_time': 1, 'c_overtime': 1}: ...
            gurobi_params = {'LogToConsole': 0, 'Threads': 32}: ...
            K = 10*3: ...
            rho = 1: ...
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
        
        # LSx
        if isinstance(weightsModel, LevelSetKDEx) or isinstance(weightsModel, LevelSetKDEx_NN):
            
             # Re-fit underlying point estimator
            weightsModel.refitPointEstimator(X_train, y_train)
            
            # Re-fit LSx
            weightsModel.fit(X_train, y_train)
            
        # If wSAA
        elif isinstance(weightsModel, RandomForestWSAA):

            # Re-fit wSAA
            weightsModel.fit(X_train, y_train)
            
            


        # Initialize
        results = pd.DataFrame()

        # For each area 
        for area in list(set(areas_test)):

            # Initialize
            results_ = {}

            # Progress
            if print_status:
                print('===========================================================================================')
                print('# Area:', area)

            # Timer
            weights_st_exec = time.time()
            weights_st_cpu = time.process_time() 

            # Select data for current area
            y_train_, y_test_ = y_train[area == areas_train], y_test[area == areas_test]
            X_train_, X_test_ = X_train[area == areas_train], X_test[area == areas_test]

            # Number of patient cases to schedule
            M = len(y_test_)

            # Set time budget (based on utilization-adjusted historical duration per area x number of cases on test day)
            T = (hist_durations[area] / rho) * M 
            
            # SAA
            if isinstance(weightsModel, SampleAverageApproximation):
                
                # Fit SAA
                weightsModel.fit(y_train_)
                
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
                c_waiting_time, c_overtime = cost_params_['c_waiting_time'], cost_params_['c_overtime']

                # Initialize optimization model
                ddps = PatientScheduling(c_waiting_time=c_waiting_time, c_overtime=c_overtime, T=T, **gurobi_params)
                
                # Set up optimization model with scenarios
                ddps.create(z=scenarios)

                # Solve
                schedule, times, status, solutions, gap = ddps.optimize()

                # Waiting time
                scheduled_start, scheduled_duration, actual_duration, waiting_time = [], [], [], []    

                for j in range(M):

                    scheduled_start += [schedule[j]]
                    scheduled_duration += [times[j]]
                    actual_duration += [y_test_[j]]
                    waiting_time += [0] if j == 0 else (
                        [max([0]+[waiting_time[j-1]+actual_duration[j-1]-scheduled_duration[j-1]])])
         
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
                    'c_waiting_time': c_waiting_time,
                    'c_overtime': c_overtime,
                    'historical_time_budget': hist_durations[area] * M,
                    'utilization': rho,
                    'assigned_time_budget': T,
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
            print('===========================================================================================')
            print('>>>> Done in', dt.datetime.now().replace(microsecond=0) - start_time)

        return results



    

    

    #### Function to run data-driven patient scheduling experiment by room
    def run_by_room(self, X, y, date, dates, areas, rooms, weightsModel = None, 
                    cost_params = {'c_waiting_time': 1, 'c_overtime': 1}, 
                    gurobi_params = {'LogToConsole': 0, 'Threads': 32}, 
                    K = 10**3, rho = 1, print_status = False, **kwargs):



        """

        ...

        Arguments:

            X: np.array(n,p)
            y: np.array(n,)
            date: ...
            dates: np.array(n,)
            areas: np.array(n,)
            rooms: np.array(n,)
            weightsModel = None: ...
            cost_params = {'c_waiting_time': 1, 'c_overtime': 1}: ...
            gurobi_params = {'LogToConsole': 0, 'Threads': 32}: ...
            K = 10*3: ...
            rho = 1: ...
            print_status = True: ...

        Returns:

            results(pd.DataFrame): ...


        """

        ## Pre-processing

        # Pre-process inputs
        X = np.array(X)
        y = np.array(y).flatten()
        dates = np.array(dates).flatten()
        areas = np.array(areas).flatten()
        rooms = np.array(rooms).flatten()

        # Check inputs
        inputs = {'X': X, 'y': y, 'dates': dates, 'areas': areas, 'rooms': rooms}
        for input_ in inputs:
            if not len(inputs[input_]) == len(y):
                raise ValueError('Inputs X, y, dates, areas, and rooms need to have same length.')

        # Train-test split
        X_train, X_test = X[dates < date], X[dates == date]
        y_train, y_test = y[dates < date].flatten(), y[dates == date].flatten()
        dates_train, dates_test = dates[dates < date], dates[dates == date]
        areas_train, areas_test = areas[dates < date], areas[dates == date]
        rooms_train, rooms_test = rooms[dates < date], rooms[dates == date]


        # Get time budgets per area
        time_budgets = self.get_time_budget(y_train, areas_train, rho, which = kwargs.get('which', 'median'))



        ## Fit weights models
        SAA_by_area = False

        # LSx
        if isinstance(weightsModel, LevelSetKDEx) or isinstance(weightsModel, LevelSetKDEx_NN):

            # Fit underlying point estimator
            weightsModel.refitPointEstimator(X_train, y_train)

            # Fit LSx
            weightsModel.fit(X_train, y_train)

        # wSAA
        elif isinstance(weightsModel, RandomForestWSAA):

            # Fit wSAA
            weightsModel.fit(X_train, y_train)

        # SAA by area
        elif isinstance(weightsModel, SampleAverageApproximation):

            weightsModels = {}
            SAA_by_area = True

            # For each area 
            for area in list(set(areas_test)):

                # Fit copy of SAA
                weightsModels[area] = copy.deepcopy(weightsModel)
                weightsModels[area].fit(y_train[area == areas_train])

        # SAA
        elif weightsModel is None:

            # Fit SAA
            weightsModel = SampleAverageApproximation()
            weightsModel.fit(y_train)


        ## Application

        # Initialize
        results = pd.DataFrame()

        # For each room 
        for room in list(set(rooms_test)):

            # Initialize
            results_ = []

            # Progress
            if print_status:
                print('======================================================================================')
                print('# Room:', room)

            # Timer
            weights_st_exec = time.time()
            weights_st_cpu = time.process_time() 

            # Select test data for current room
            X_test_ = X_test[room == rooms_test]
            y_test_ = y_test[room == rooms_test]
            areas_test_ = areas_test[room == rooms_test]
            rooms_test_ = rooms_test[room == rooms_test]

            # Number of patient cases to schedule
            M = len(y_test_)

            # Set total time budget
            T = 0
            for j in range(M):
                T += time_budgets[areas_test_[j]]

            # SAA by area
            if SAA_by_area:

                # For each area
                conditionalDistributions = {}
                for area in list(set(areas_test_)):

                    # Get estimated conditional distribution by area
                    conditionalDistributions[area] = weightsModels[area].getWeights(outputType='summarized')

                # For each patient 
                conditionalDistribution = []
                for j in range(M):

                    # Get estimated conditional distribution by patient
                    conditionalDistribution += copy.deepcopy(conditionalDistributions[areas_test_[j]])

            else:

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
                    print('## Cost param setting:', cost_params_)

                # Timer
                opt_st_exec = time.time()
                opt_st_cpu = time.process_time() 

                # Optimization
                c_waiting_time, c_overtime = cost_params_['c_waiting_time'], cost_params_['c_overtime']

                # Initialize optimization model
                ddps = PatientScheduling(c_waiting_time=c_waiting_time, c_overtime=c_overtime, T=T, **gurobi_params)

                # Set up optimization model with scenarios
                ddps.create(z=scenarios)

                # Solve
                schedule, times, status, solutions, gap = ddps.optimize()

                # Waiting time
                scheduled_start, scheduled_duration, actual_duration, waiting_time = [], [], [], []    

                for j in range(M):

                    scheduled_start += [schedule[j]]
                    scheduled_duration += [times[j]]
                    actual_duration += [y_test_[j]]
                    waiting_time += [0] if j == 0 else (
                        [max([0]+[waiting_time[j-1]+actual_duration[j-1]-scheduled_duration[j-1]])])

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
                    'area': areas_test_[0] if len(set(areas_test_)) == 1 else areas_test_,
                    'room': int(room),
                    'n_patients': M,
                    'n_scenarios': K,
                    'c_waiting_time': c_waiting_time,
                    'c_overtime': c_overtime,
                    'utilization': rho,
                    'assigned_time_budget': T,
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
                results_ += [copy.deepcopy(result)]   

            # Create data frame results
            results = pd.concat([results, pd.DataFrame(results_)])    

        # Progress
        if print_status:
            print('======================================================================================')
            print('>>>> Done in', dt.datetime.now().replace(microsecond=0) - start_time)

        return results


    
    
    
    
    
    #### Function to run data-driven patient scheduling experiment
    def runByRoom(
        self, X, y, date, dates, areas, rooms, weightsModel = None, 
        K = [10**3], rho = [1.00], cost_params = [{'c_waiting_time': 1, 'c_overtime': 1}], 
        gurobi_params = {'LogToConsole': 0, 'Threads': 32}, print_status = False, **kwargs):



        """

        ...

        Arguments:

            X: np.array(n,p)
            y: np.array(n,)
            date: ...
            dates: np.array(n,)
            areas: np.array(n,)
            rooms: np.array(n,)
            weightsModel = None: ...
            K = [10*3]: ...
            rho = [1.00]: ...
            cost_params = [{'c_waiting_time': 1, 'c_overtime': 1}]: ...
            gurobi_params = {'LogToConsole': 0, 'Threads': 32}: ...
            print_status = True: ...
            kwargs: ...

        Returns:

            results(pd.DataFrame): ...


        """

        ## Pre-processing

        # Pre-process inputs
        X = np.array(X)
        y = np.array(y).flatten()
        dates = np.array(dates).flatten()
        areas = np.array(areas).flatten()
        rooms = np.array(rooms).flatten()

        # Check inputs

        inputs = {'X': X, 'y': y, 'dates': dates, 'areas': areas, 'rooms': rooms}

        for input_ in inputs:
            if not len(inputs[input_]) == len(y):
                raise ValueError('Inputs X, y, dates, areas, and rooms need to have same length.')

        if type(K) != list or type(rho) != list or type(cost_params) != list:
            raise ValueError('K, rho, and cost_params have to be lists.')

        # Timer
        start_time = dt.datetime.now().replace(microsecond=0)


        # Train-test split
        X_train, X_test = X[dates < date], X[dates == date]
        y_train, y_test = y[dates < date].flatten(), y[dates == date].flatten()
        dates_train, dates_test = dates[dates < date], dates[dates == date]
        areas_train, areas_test = areas[dates < date], areas[dates == date]
        rooms_train, rooms_test = rooms[dates < date], rooms[dates == date]


        ## Fit weights models

        # Timer
        fit_weightsModel_st_exec = time.time()
        fit_weightsModel_st_cpu = time.process_time()    

        # Indicator
        SAA_by_area = False

        # LSx
        if isinstance(weightsModel, LevelSetKDEx) or isinstance(weightsModel, LevelSetKDEx_NN):

            # Fit underlying point estimator
            weightsModel.refitPointEstimator(X_train, y_train)

            # Fit LSx
            weightsModel.fit(X_train, y_train)

        # wSAA
        elif isinstance(weightsModel, RandomForestWSAA):

            # Fit wSAA
            weightsModel.fit(X_train, y_train)

        # SAA by area
        elif isinstance(weightsModel, SampleAverageApproximation):

            weightsModels = {}
            SAA_by_area = True

            # For each area 
            for area in list(set(areas_test)):

                # Fit copy of SAA
                weightsModels[area] = copy.deepcopy(weightsModel)
                weightsModels[area].fit(y_train[area == areas_train])

        # SAA
        elif weightsModel is None:

            # Fit SAA
            weightsModel = SampleAverageApproximation()
            weightsModel.fit(y_train)

        # Timer
        fit_weightsModel_exec_time_sec = time.time()-fit_weightsModel_st_exec
        fit_weightsModel_cpu_time_sec = time.process_time()-fit_weightsModel_st_cpu




        ## Application

        # Initialize
        results = []

        # For each room 
        for room in list(set(rooms_test)): 

            # Select test data for current room
            X_test_ = X_test[room == rooms_test]
            y_test_ = y_test[room == rooms_test]
            areas_test_ = areas_test[room == rooms_test]
            rooms_test_ = rooms_test[room == rooms_test]

            # Number of patient cases to schedule
            M = len(y_test_)



            ## Generate conditional distributions

            # Timer
            apply_weightsModel_st_exec = time.time()
            apply_weightsModel_st_cpu = time.process_time()    

            # SAA by area
            if SAA_by_area:

                # For each area
                conditionalDistributions = {}
                for area in list(set(areas_test_)):

                    # Get estimated conditional distribution by area
                    conditionalDistributions[area] = weightsModels[area].getWeights(outputType='summarized')

                # For each patient 
                conditionalDistribution = []
                for j in range(M):

                    # Get estimated conditional distribution by patient
                    conditionalDistribution += copy.deepcopy(conditionalDistributions[areas_test_[j]])

            else:

                # Get estimated conditional distribution
                conditionalDistribution = weightsModel.getWeights(X_test_, outputType='summarized')

            # Timer
            apply_weightsModel_exec_time_sec = time.time()-apply_weightsModel_st_exec
            apply_weightsModel_cpu_time_sec = time.process_time()-apply_weightsModel_st_cpu




            # For each number of scenarios
            for K_ in K:


                ## Generate K scenarios

                # Timer
                gen_scenarios_st_exec = time.time()
                gen_scenarios_st_cpu = time.process_time() 

                # Initialize
                scenarios = []

                # Draw
                for k in range(K_):    

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
                gen_scenarios_exec_time_sec = time.time()-gen_scenarios_st_exec
                gen_scenarios_cpu_time_sec = time.process_time()-gen_scenarios_st_cpu




                # For each utilization
                for rho_ in rho:

                    # Get time budgets per area
                    time_budgets = self.get_time_budget(y_train, areas_train, rho_, which = kwargs.get('which', 'median'))

                    # Set total time budget
                    T = 0
                    for j in range(M):
                        T += time_budgets[areas_test_[j]]


                    # For each cost param setting
                    for cost_params_ in cost_params:

                        # Timer
                        opt_st_exec = time.time()
                        opt_st_cpu = time.process_time() 

                        # Optimization
                        c_waiting_time, c_overtime = cost_params_['c_waiting_time'], cost_params_['c_overtime']
                        
                        # Progress
                        if print_status:
                            print(
                                '## Room:', room, 
                                '| Scenarios:', K_, 
                                '| Utilization:', rho_, 
                                '| Overtime cost:', c_overtime
                            )

                        # Initialize optimization model
                        ddps = PatientScheduling(
                            c_waiting_time=c_waiting_time, 
                            c_overtime=c_overtime, 
                            T=T, 
                            **gurobi_params
                        )

                        # Set up optimization model with scenarios
                        ddps.create(z=scenarios)

                        # Solve
                        schedule, times, status, solutions, gap = ddps.optimize()

                        # Waiting time
                        scheduled_start, scheduled_duration, actual_duration, waiting_time = [], [], [], []    

                        for j in range(M):

                            scheduled_start += [schedule[j]]
                            scheduled_duration += [times[j]]
                            actual_duration += [y_test_[j]]
                            waiting_time += [0] if j == 0 else [max(
                                [0]+[waiting_time[j-1]+actual_duration[j-1]-scheduled_duration[j-1]])]

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
                            'area': areas_test_[0] if len(set(areas_test_)) == 1 else areas_test_,
                            'room': int(room),
                            'n_patients': M,
                            'n_scenarios': K_,
                            'c_waiting_time': c_waiting_time,
                            'c_overtime': c_overtime,
                            'utilization': rho_,
                            'assigned_time_budget': T,
                            'waiting_time': sum(waiting_time),
                            'overtime': overtime,
                            'cost': cost,
                            'scheduled_total_durations': sum(scheduled_duration),
                            'actual_total_durations': sum(actual_duration),
                            'optimization_exec_time_sec': optimization_exec_time_sec,
                            'optimization_cpu_time_sec': optimization_cpu_time_sec,
                            'scenarios_exec_time_sec': gen_scenarios_exec_time_sec,
                            'scenarios_cpu_time_sec': gen_scenarios_cpu_time_sec,
                            'weights_exec_time_sec': fit_weightsModel_exec_time_sec+apply_weightsModel_exec_time_sec,
                            'weights_cpu_time_sec': fit_weightsModel_cpu_time_sec+apply_weightsModel_cpu_time_sec

                        }

                        # Add to results
                        results += [copy.deepcopy(result)]   


        # Create data frame results
        results = pd.concat([results, pd.DataFrame(results_)])    

        # Progress
        if print_status:
            print('## >>>> Done in', dt.datetime.now().replace(microsecond=0) - start_time)

        return results
   





    #### Function to run data-driven patient scheduling experiment
    def run_xArea_xRoom(
        self, X, y, date, dates, areas, rooms, weightsModel = None, 
        K = [10**3], rho = [1.00], cost_params = [{'c_waiting_time': 1, 'c_overtime': 1}], 
        gurobi_params = {'LogToConsole': 0, 'Threads': 32}, print_status = False, **kwargs):



        """

        ...

        Arguments:

            X: np.array(n,p)
            y: np.array(n,)
            date: ...
            dates: np.array(n,)
            areas: np.array(n,)
            rooms: np.array(n,)
            weightsModel = None: ...
            K = [10*3]: ...
            rho = [1.00]: ...
            cost_params = [{'c_waiting_time': 1, 'c_overtime': 1}]: ...
            gurobi_params = {'LogToConsole': 0, 'Threads': 32}: ...
            print_status = True: ...
            kwargs: ...

        Returns:

            results(pd.DataFrame): ...


        """

        ## Pre-processing

        # Pre-process inputs
        X = np.array(X)
        y = np.array(y).flatten()
        dates = np.array(dates).flatten()
        areas = np.array(areas).flatten()
        rooms = np.array(rooms).flatten()

        # Check input data
        inputs = {'X': X, 'y': y, 'dates': dates, 'areas': areas, 'rooms': rooms}
        for input_ in inputs:
            if not len(inputs[input_]) == len(y):
                raise ValueError('Inputs X, y, dates, areas, and rooms need to have same length.')

        # Check input params
        if type(K) != list or type(rho) != list or type(cost_params) != list:
            raise ValueError('K, rho, and cost_params have to be lists.')

        # Timer
        start_time = dt.datetime.now().replace(microsecond=0)


        # Train-test split
        X_train, X_test = X[dates < date], X[dates == date]
        y_train, y_test = y[dates < date].flatten(), y[dates == date].flatten()
        dates_train, dates_test = dates[dates < date], dates[dates == date]
        areas_train, areas_test = areas[dates < date], areas[dates == date]
        rooms_train, rooms_test = rooms[dates < date], rooms[dates == date]


        ## Fit weights models

        # Timer
        fit_weightsModel_st_exec = time.time()
        fit_weightsModel_st_cpu = time.process_time()    

        # Indicator
        SAA_by_area = False

        # LSx
        if isinstance(weightsModel, LevelSetKDEx) or isinstance(weightsModel, LevelSetKDEx_NN):

            # Fit underlying point estimator
            weightsModel.refitPointEstimator(X_train, y_train)

            # Fit LSx
            weightsModel.fit(X_train, y_train)

        # wSAA
        elif isinstance(weightsModel, RandomForestWSAA):

            # Fit wSAA
            weightsModel.fit(X_train, y_train)

        # SAA by area
        elif isinstance(weightsModel, SampleAverageApproximation):

            weightsModels = {}
            SAA_by_area = True

            # For each area 
            for area in list(set(areas_test)):

                # Fit copy of SAA
                weightsModels[area] = copy.deepcopy(weightsModel)
                weightsModels[area].fit(y_train[area == areas_train])

        # SAA
        elif weightsModel is None:

            # Fit SAA
            weightsModel = SampleAverageApproximation()
            weightsModel.fit(y_train)

        # Timer
        fit_weightsModel_exec_time_sec = time.time()-fit_weightsModel_st_exec
        fit_weightsModel_cpu_time_sec = time.process_time()-fit_weightsModel_st_cpu




        ## Application

        # Initialize
        results = []

        # For each area
        for area in list(set(areas_test)):

            # For each room 
            for room in list(set(rooms_test[area == areas_test])): 

                # Select test data for current room
                sel = (area == areas_test) & (room == rooms_test)
                X_test_, y_test_, areas_test_, rooms_test_ = X_test[sel], y_test[sel], areas_test[sel], rooms_test[sel]

                # Number of patient cases to schedule
                M = len(y_test_)



                ## Generate conditional distributions

                # Timer
                apply_weightsModel_st_exec = time.time()
                apply_weightsModel_st_cpu = time.process_time()    

                # SAA by area
                if SAA_by_area:

                    # For each area
                    conditionalDistributions = {}
                    for area in list(set(areas_test_)):

                        # Get estimated conditional distribution by area
                        conditionalDistributions[area] = weightsModels[area].getWeights(outputType='summarized')

                    # For each patient 
                    conditionalDistribution = []
                    for j in range(M):

                        # Get estimated conditional distribution by patient
                        conditionalDistribution += copy.deepcopy(conditionalDistributions[areas_test_[j]])

                else:

                    # Get estimated conditional distribution
                    conditionalDistribution = weightsModel.getWeights(X_test_, outputType='summarized')

                # Timer
                apply_weightsModel_exec_time_sec = time.time()-apply_weightsModel_st_exec
                apply_weightsModel_cpu_time_sec = time.process_time()-apply_weightsModel_st_cpu




                # For each number of scenarios
                for K_ in K:


                    ## Generate K scenarios

                    # Timer
                    gen_scenarios_st_exec = time.time()
                    gen_scenarios_st_cpu = time.process_time() 

                    # Initialize
                    scenarios = []

                    # Draw
                    for k in range(K_):    

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
                    gen_scenarios_exec_time_sec = time.time()-gen_scenarios_st_exec
                    gen_scenarios_cpu_time_sec = time.process_time()-gen_scenarios_st_cpu




                    # For each utilization
                    for rho_ in rho:

                        # Get time budgets per area
                        time_budgets = self.get_time_budget(
                            y_train, areas_train, rho_, which = kwargs.get('which', 'median'))

                        # Set total time budget
                        T = 0
                        for j in range(M):
                            T += time_budgets[areas_test_[j]]


                        # For each cost param setting
                        for cost_params_ in cost_params:

                            # Timer
                            opt_st_exec = time.time()
                            opt_st_cpu = time.process_time() 

                            # Optimization
                            c_waiting_time, c_overtime = cost_params_['c_waiting_time'], cost_params_['c_overtime']

                            # Progress
                            if print_status:
                                print('## Area:', area, 'Room:', room, '| Scenarios:', K_, 
                                      '| Utilization:', rho_, '| Overtime cost:', c_overtime)

                            # Initialize optimization model
                            ddps = PatientScheduling(c_waiting_time=c_waiting_time, 
                                                     c_overtime=c_overtime, T=T, **gurobi_params)

                            # Set up optimization model with scenarios
                            ddps.create(z=scenarios)

                            # Solve
                            schedule, times, status, solutions, gap = ddps.optimize()

                            # Waiting time
                            scheduled_start, scheduled_duration, actual_duration, waiting_time = [], [], [], []    

                            for j in range(M):

                                scheduled_start += [schedule[j]]
                                scheduled_duration += [times[j]]
                                actual_duration += [y_test_[j]]
                                waiting_time += [0] if j == 0 else [max(
                                    [0]+[waiting_time[j-1]+actual_duration[j-1]-scheduled_duration[j-1]])]

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
                                'room': int(room),
                                'n_patients': M,
                                'n_scenarios': K_,
                                'c_waiting_time': c_waiting_time,
                                'c_overtime': c_overtime,
                                'utilization': rho_,
                                'assigned_time_budget': T,
                                'waiting_time': sum(waiting_time),
                                'overtime': overtime,
                                'cost': cost,
                                'scheduled_total_durations': sum(scheduled_duration),
                                'actual_total_durations': sum(actual_duration),
                                'optimization_exec_time_sec': optimization_exec_time_sec,
                                'optimization_cpu_time_sec': optimization_cpu_time_sec,
                                'scenarios_exec_time_sec': gen_scenarios_exec_time_sec,
                                'scenarios_cpu_time_sec': gen_scenarios_cpu_time_sec,
                                'weights_exec_time_sec': fit_weightsModel_exec_time_sec+apply_weightsModel_exec_time_sec,
                                'weights_cpu_time_sec': fit_weightsModel_cpu_time_sec+apply_weightsModel_cpu_time_sec
                            
                            }

                            # Add to results
                            results += [copy.deepcopy(result)]   


        # Create data frame results
        results = pd.DataFrame(results)

        # Progress
        if print_status:
            print('## >>>> Done in', dt.datetime.now().replace(microsecond=0) - start_time)

        return results




    
    @contextlib.contextmanager
    def tqdm_joblib(self, tqdm_object):

        """

        Context manager
        
        Credits: 'https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution'

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



            
            
            
            


#### ...
class Evaluation:

    """
    ...
    
    """

    ## Initialize
    def __init__(self, **kwargs):

        """
        ...
        
        """
        
        return None



    #### Function to load evaluation data
    def load_data(self, **kwargs):


        return None

    
    
    
    #### Function to pre-process evaluation data
    def preprocess_data(self, **kwargs):

        """

        ...


        """

        return None