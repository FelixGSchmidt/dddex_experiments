

from multiprocessing import Pool
from functools import partial
from Optimize_kERM_functions import *

def calcProfitForConfig(T, fj, aij, ci, cvtrain_len, cvtest_len, cvtrain_KernelW, cvtest_KernelW, cvtrain_Demand, cvtest_Demand, param, k ):  

    constants = T, fj[k], aij[k], ci[k]
    
    c0_val = [(aij[k][0][0])*param,
              (aij[k][1][1])*param,
              (aij[k][2][2])*param];

    u, b = learnModel(cvtrain_len, cvtrain_KernelW, cvtrain_Demand, c0_val, constants) 

    q = prescribe(u, b, cvtrain_len, cvtest_len, cvtest_KernelW, c0_val)

    return evaluateProfit(cvtest_len, q, cvtest_Demand, constants)



def evaluateProfitForConfigs(T, fj, aij, ci, cvtrain_len, cvtest_len, cvtrain_KernelW, cvtest_KernelW, cvtrain_Demand, cvtest_Demand, param, numConfigLines):

    agents = 24

    calcProfitSimple = partial(calcProfitForConfig, T, fj, aij, ci, cvtrain_len, cvtest_len, cvtrain_KernelW, cvtest_KernelW, cvtrain_Demand, cvtest_Demand, param )

    mypool = Pool(processes=agents)

    profit_param = mypool.map(calcProfitSimple, list(range(numConfigLines)))

    mypool.terminate();

    return profit_param;

