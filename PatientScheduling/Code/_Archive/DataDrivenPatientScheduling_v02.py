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

    
    
    def room_assignment(self, data, max_rooms = np.inf, max_room_capacity = None, 
                        room_utilization = 1, by_area=True, **kwargs):
        
        """
        
        ...
        
        """
    
        # If no max room capacity is given
        if max_room_capacity is None:

            # Max room capacity (based on historical capacities)
            max_room_capacity = np.median(data.loc[data.train_test == 'train'].groupby('date').agg(
                capacity=('room_capacity', lambda x: x.iloc[0])).reset_index().capacity)

            # Max room capacity adjusted with utilization
            max_room_capacity = max_room_capacity / room_utilization

        # Historical duration by area defined as median in-sample daily durations
        historical_durations = data.loc[data.train_test == 'train'].groupby('area').agg(
            median_duration = ('duration', np.median)).reset_index()

        # Test dates
        dates = data.date.unique()

        # Initialize
        room_assignments = []

        # For each test date
        for date in dates:

            # By area
            if by_area:

                # Areas 
                areas = sorted(data.loc[data.date == date].area.unique())

                # Room counter
                room_counter = 0

                # For each area
                for area in areas:

                    # Select data
                    patients = data.loc[(data.date == date) & (data.area == area)].patient_id
                    n_patients = len(patients)
                    durations = historical_durations.loc[historical_durations.area == area].median_duration.item()

                    # Assign rooms 
                    rooms = self.assign_rooms_(max_room_capacity, max_rooms, n_patients, durations)

                    # Adjust for rooms already asigned
                    rooms = list(np.array(rooms) + room_counter)

                    # Update room counter
                    room_counter = max(rooms)

                    # Store
                    room_assignments += [pd.DataFrame(dict(date=[date]*n_patients, 
                                                           area=[area]*n_patients, 
                                                           patient_id=patients, 
                                                           room=rooms))]

            # Independent of area
            else:

                # Select data
                areas = data.loc[data.date == date].area
                patients = data.loc[data.date == date].patient_id
                n_patients = len(patients)
                durations = []
                for area in areas:
                    durations += [historical_durations.loc[historical_durations.area == area].median_duration.item()]

                # Room counter
                room_counter = 0

                # Assign rooms 
                rooms = self.assign_rooms_(max_room_capacity, max_rooms, n_patients, durations)

                # Adjust for rooms already asigned
                rooms = list(np.array(rooms) + room_counter)

                # Update room counter
                room_counter = max(rooms)

                # Store
                room_assignments += [pd.DataFrame(dict(date=[date]*n_patients, 
                                                       area=areas,
                                                       patient_id=patients, 
                                                       room=rooms))]

        # Finalize
        room_assignments = pd.concat(room_assignments)

        return room_assignments


    
    
    
    def assign_rooms_(self, max_room_capacity, max_rooms, n_patients, durations, **kwargs):
        
        """
        
        ...
        
        
        """
    
        # Inputs
        if type(durations) == list:
            if not len(durations) == n_patients:

                raise ValueError('Length of durations should be n_patients')

        else:
            durations = [durations] * n_patients

        # Initialize
        current_capacity = 0
        room = 1
        rooms = []

        # For all patients
        for patient in range(n_patients):

            # Empty rooms are available
            if room < max_rooms:          

                # Current room has capacity available
                if current_capacity + durations[patient] <= max_room_capacity:

                    # Add to rooms
                    rooms += [room]

                    # Update current capacity
                    current_capacity += durations[patient]

                # Current room has no capacity available
                else:

                    # Reset current capacity
                    current_capacity = 0

                    # New room
                    room += 1

                    # Add to rooms
                    rooms += [room]

                    # Update current capacity
                    current_capacity += durations[patient]

            # No empty rooms are available
            else:

                # Overfill rooms            
                rooms += [(max_rooms + room) % max_rooms + 1]

                # Update shadow room counter
                room += 1

        return rooms
    
    
    
    
    
    

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
    def run(self, X, y, date, dates, areas, rooms, weightsModel = None, 
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
            sel = (room == rooms_test)
            X_test_, y_test_, areas_test_, rooms_test_ = X_test[sel], y_test[sel], areas_test[sel], rooms_test[sel]

            # Number of patient cases to schedule
            M = len(y_test_)

            ## Generate conditional distributions

            # Timer
            apply_weightsModel_st_exec = time.time()
            apply_weightsModel_st_cpu = time.process_time()    

            # SAA by area
            if isinstance(weightsModel, dict):

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
                            print('## Room:', room, 
                                  '| Scenarios:', K_, 
                                  '| Utilization:', rho_, 
                                  '| Overtime cost:', c_overtime)

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
                            'area': areas_test_.item() if len(areas_test_) == 1 else set(areas_test_),
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
    
    
    
    
    
    def plot_medians_(self, grid, x_var, y_var, hue_var, **kwargs):
        
        """
        
        ...
        
        
        """
    
        # Defaults
        linewidth = 1.0
        width=0.66
        fliersize=0
        whis=0

        # Box-plots
        grid.map_dataframe(
            sns.boxplot, x=x_var, y=y_var, hue=hue_var, 
            width=kwargs.get('width', width), fliersize=kwargs.get('fliersize', fliersize), 
            whis=kwargs.get('whis', whis), linewidth=kwargs.get('linewidth', linewidth), 
            palette=kwargs.get('palette', 'magma'))

        # Box-plot sizes
        fac = 0.75

        # iterating through Axes instances
        for ax in grid.axes.flatten():

            # iterating through axes artists:
            for c in ax.get_children():

                # searching for PathPatches
                if isinstance(c, PathPatch):

                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    xmax = np.max(verts_sub[:, 0])
                    xmid = 0.5*(xmin+xmax)
                    xhalf = 0.5*(xmax - xmin)

                    # setting new width of box
                    xmin_new = xmid-fac*xhalf
                    xmax_new = xmid+fac*xhalf
                    verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                    verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                    # setting new width of median line
                    for l in ax.lines:
                        #if np.all(l.get_xdata() == [xmin, xmax]):
                        if (l.get_xdata()[0] == xmin) & (l.get_xdata()[1] == xmax):
                            l.set_xdata([xmin_new, xmax_new])
                            l.set_linewidth(3.0)
                            l.set_color('red')


        return grid
    
    
    
    
    
    
    
    def plot_totals_(self, grid, x_var, y_var, hue_var, **kwargs):
        
        """
        
        ...
        
        
        """
    
        # Point plots
        grid.map_dataframe(
            sns.pointplot, x=x_var, y=y_var, hue=hue_var, 
            dodge=0.5, join=False, markers='x',
            palette=kwargs.get('palette', 'magma'))





        return grid
    
    
    
    

    

    def plot_prescriptive_performance(self, plotData, grid_var, x_var, y_var, hue_var, 
                                      kind='medians', facet_h=6, facet_w=3, **kwargs):
        
        """
        
        ...
        
        
        """

        # Defaults
        linewidth = 1.0
        xlabel='overtime penalty'
        ylabel='cost relative to SAA'
        ylim=(-0.05, 1.85)

        # Setup facet grid
        grid = sns.FacetGrid(plotData, col=grid_var, height=facet_h, aspect=facet_w/facet_h)

        # Box-plots
        if kind=='medians':

            grid = self.plot_medians_(grid, x_var, y_var, hue_var, **kwargs)

        elif kind=='totals':

            grid = self.plot_totals_(grid, x_var, y_var, hue_var, **kwargs)

        else:

            return None


        # Reference line
        grid.refline(y=1.0, color='red', linewidth=kwargs.get('linewidth', linewidth), linestyle='--')

        # Facet borders
        for ax in grid.axes.flatten(): 
            for _, spine in ax.spines.items():
                spine.set_visible(True) 
                spine.set_color('black')
                spine.set_linewidth(1.0)

        # Axis labels
        for ax in grid.axes.flatten():

            ax.set_xlabel(xlabel=kwargs.get('xlabel', xlabel), fontsize=10, fontweight='bold')
            ax.set_ylabel(ylabel=kwargs.get('ylabel', ylabel), fontsize=10, fontweight='bold')
            ax.title.set_size(10)
            ax.title.set_weight('bold')
            ax.xaxis.set_label_position('bottom')


        # Legend
        grid.add_legend(ncol=1, loc='right')

        # Limits
        grid.set(ylim=kwargs.get('ylim', ylim))

        return grid.figure