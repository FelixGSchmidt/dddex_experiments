
# Learning using kERM approach


import gurobipy as gb
import csv
import numpy.random
import math
from Optimize_kERM_functions import *
from Optimize_kERM_parallel import *


def paramTuneAndLearnModelkERM(inputDataset,  ## input demand dataset
                               configFile, ## config file with configurations for training
                               outputDataset, ## write prescription functions to file
                               outputConfigFile, ##write configurations found to file
                               c0_params, ## array of c0 params for hyperparam tuning to be scaled with aij
                               gl_m, # max number of training samples
                               gl_m_cv # number of training samples for CV
                               ):

    #const
    T = 5 # number of days in period
    

    maxNumConfigs = 1000;

    gl_Demand = [[[0.0 for z in range(3)] for y in range(T)] for x in range(gl_m)] #Demand[k][t][i]

    aij = [[[0.0 for y in range(3)] for x in range(3)] for z in range(maxNumConfigs)];
    ci = [[0.0 for x in range(3)] for y in range(maxNumConfigs)];
    pi = [[0.0 for x in range(3)] for y in range(maxNumConfigs)];
    fj = [[0.0 for x in range(3)] for y in range(maxNumConfigs)];
    vj = [[0.0 for x in range(3)] for y in range(maxNumConfigs)];

    #### READ CONFIG FILE
    lineID=0;
    with open(configFile) as csvfile:
        myreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in myreader:
            if lineID>0: #skip first line: column names
                for j in range(3):
                    fj[lineID-1][j] = float(row[0+j].replace(',','.'))
                    vj[lineID-1][j] = float(row[3+j].replace(',','.'))
                    pi[lineID-1][j] = float(row[6+j].replace(',','.'))
                    ci[lineID-1][j] = float(row[9+j].replace(',','.'))
            lineID += 1

    numConfigLines = lineID-1;

    for k in range(numConfigLines):
        for i in range(3):
            for j in range(3):
                aij[k][i][j]= pi[k][i] - vj[k][j] + ci[k][i];


    dataArrayLen = 0
    k = 0

    #Read demand
    with open(inputDataset) as csvfile:
        myreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in myreader:
            if dataArrayLen>0: #skip first line: column names

                for t in range(T):
                    for i in range(3):
                        gl_Demand[k][t][i] = float(row[1 + t + i*T].replace(',', '.')) #demand data

                k = k + 1

            dataArrayLen += 1

    #correct for data array len (skipped first line)
    dataArrayLen -= 1



    try:

        KernelW = getRF_Kernel(gl_m, gl_m)

        cvtest_selector = list(range(gl_m_cv,gl_m));
        cvtrain_selector = list(range(gl_m_cv)); 

        cvtrain_len=len(cvtrain_selector)
        cvtest_len=len(cvtest_selector)
 
        cvtrain_Demand = [[[0.0 for z in range(3)] for y in range(T)] for x in range(cvtrain_len)] #Demand[k][t][i]
        cvtest_Demand = [[[0.0 for z in range(3)] for y in range(T)] for x in range(cvtest_len)] #Demand[k][t][i]
        cvtrain_KernelW = [[0.0 for y in range(cvtrain_len)] for x in range(cvtrain_len)] #KernelW[k1,k2]
        cvtest_KernelW = [[0.0 for y in range(cvtest_len)] for x in range(cvtrain_len)] #KernelW[k1,k2]

        nparam = len(c0_params)

        profit_param = [[0.0 for x in range(nparam)] for k in range(numConfigLines)] #profit_param[k][i]

        iMaxVal = [0 for k in range(numConfigLines)]

        if True: # tune parameters
            for i in range(nparam):

                #prepare sub-kernels for training and testing
                for k in range(cvtrain_len):
                    cvtrain_Demand[k] = gl_Demand[cvtrain_selector[k]]
                    for j in range(cvtrain_len):
                        cvtrain_KernelW[k][j] = KernelW[cvtrain_selector[k]][cvtrain_selector[j]]
                    for j in range(cvtest_len):
                        cvtest_KernelW[k][j] = KernelW[cvtrain_selector[k]][cvtest_selector[j]]


                for k in range(cvtest_len):
                    cvtest_Demand[k] = gl_Demand[cvtest_selector[k]]

                #parallel eval
                print("Starting parallel profit evaluation...")
                profit_k = evaluateProfitForConfigs(T, fj, aij, ci, cvtrain_len, cvtest_len, cvtrain_KernelW, cvtest_KernelW, cvtrain_Demand, cvtest_Demand, c0_params[i], numConfigLines);
                for k in range(numConfigLines):
                    profit_param[k][i] =float(profit_k[k])

                    print("Found profit: ", profit_param[k][i])

                    if(profit_param[k][i] > profit_param[k][iMaxVal[k]]):
                        iMaxVal[k] = i
                        print("Found new best profit for i=", iMaxVal[k])


        #########
        # Choose configuration and learn model on full dataset
        #########

        u = [[[0.0 for y in range(3)] for x in range(gl_m)] for z in range(maxNumConfigs)]; #u[k,j]
        b = [[0.0 for x in range(3)] for z in range(maxNumConfigs)]; #b[j]

        # learn a model for each configuration
        for k in range(numConfigLines):

            ### setting factor *gl_m/gl_m_cv for different training sample size
            c0_val = [(aij[k][0][0])*c0_params[iMaxVal[k]]*gl_m/gl_m_cv,
                      (aij[k][1][1])*c0_params[iMaxVal[k]]*gl_m/gl_m_cv,
                      (aij[k][2][2])*c0_params[iMaxVal[k]]*gl_m/gl_m_cv]
            

            KernelW = getRF_Kernel(gl_m, gl_m)

            constants = T, fj[k], aij[k], ci[k]
            u[k], b[k] = learnModel(gl_m, KernelW, gl_Demand, c0_val, constants)

        #write used configuration in csv
        with open(outputConfigFile, 'w') as csvfile:
            mywriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            mywriter.writerow(["ConfigID", "c0_1", "c0_2", "c0_3"])
            for k in range(numConfigLines):
                c0_val = [(aij[k][0][0])*c0_params[iMaxVal[k]]*gl_m/gl_m_cv,
                          (aij[k][1][1])*c0_params[iMaxVal[k]]*gl_m/gl_m_cv,
                          (aij[k][2][2])*c0_params[iMaxVal[k]]*gl_m/gl_m_cv]
                mywriter.writerow([k,  ("{:.12f}".format(c0_val[0])).replace(".",","),
                                   ("{:.12f}".format(c0_val[1])).replace(".",","), ("{:.12f}".format(c0_val[2])).replace(".",",")])

        #write results in csv
        with open(outputDataset, 'w') as csvfile:
            mywriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            mywriter.writerow(["ConfigID", "k", "j", "u_k_j", "b_j"])
            for cfgID in range(numConfigLines):
                for k in range(1,gl_m+1):
                    for j in range(1,3+1):
                        mywriter.writerow([cfgID, k, j, ("{:.12f}".format(u[cfgID][k-1][j-1])).replace(".",","), ("{:.12f}".format(b[cfgID][j-1])).replace(".",",")])

    except gb.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')


    print("kERM function tuning and training completed.")

