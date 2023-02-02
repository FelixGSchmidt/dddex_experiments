

from Optimize_kERM_ParamTune import *
from MPL_Optimize_Capacity_functions import *

import os.path

try:

    filepath = '';  # 'datasets/';


    found_problem=0;

    ## make sure all required files exist
    if not os.path.exists("kERM_Dataset.csv"):
        print("kERM_Dataset.csv does not exist!")
        found_problem=1;
    if not os.path.exists("config_param_list.csv"):
        print("config_param_list.csv does not exist!")
        found_problem=1;
    if not os.path.exists("RF_Kernel_Learning.csv"):
        print("RF_Kernel_Learning.csv does not exist!")
        found_problem=1;
    if not os.path.exists("opti_input.csv"):
        print("opti_input.csv does not exist!")
        found_problem=1;



    if found_problem==0:
        print("Starting parallel evaluations..")


        print("Starting parallel evaluations..")
	    optimizeSAA_Cap(filepath + 'opti_input.csv',
				filepath + 'opti_vals_SAA.csv',
				4, #number of years of total data
				filepath + 'config_param_list.csv',
				157) # Train size in weeks
				
		optimizeDetermCap( filepath + 'opti_input.csv',
                           filepath + 'opti_vals.csv',
                           4, #number of years of total data
                           filepath + 'config_param_list.csv')


        print("Starting Weighted SAA evaluation..")
        optimizeKernelSAA_Cap(filepath + 'opti_input.csv',
                                filepath + 'opti_vals_KernelSAA.csv',
                                4, #number of years of total data
                                filepath + 'config_param_list.csv',
                                157, # Train size in weeks
                                filepath + 'RF_Kernel_Testing.csv')

		c0_params = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                     0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25,
                     0.26, 0.27, 0.28, 0.29, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0,
                     2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 10000.0]
					 
        print("Starting kERM evaluation..")
		paramTuneAndLearnModelkERM(filepath + 'kERM_Dataset.csv', ## input demand dataset
                               filepath + 'config_param_list.csv', ## config file with configurations for training
                               filepath + 'kERM_Model.csv', ## write prescription functions to file
                               filepath + 'kERM_output_configs.csv', ##write configurations found to file
                               c0_params, ## array of c0 params for hyperparam tuning to be scaled with aij
                               157, # max number of training samples
                               105 # number of training samples for CV
                               ):

        print("All evaluations are done. Exiting..")

    else:
        print("One or more problems were found. Exiting..")


except gb.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')


