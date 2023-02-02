
# Learning using kERM approach


import gurobipy as gb
import csv
import math

#################################################################
# RF KERNEL FUNCTION
#################################################################
def getRF_Kernel(m1, m2) :

    
    filepath = '';  
    
    KernelW = [[0.0 for y in range(m2)] for x in range(m1)] #KernelW[k1,k2]
        
    filename = filepath + "RF_Kernel_Learning.csv"
    
    with open(filename) as csvfile:
        myreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        i=-1;
        for row in myreader:
            if i >= 0: ## skip first line
                for j in range(m1):
                    KernelW[i][j] = float(row[j].replace(',','.'))
            i = i+1;
    
    return KernelW
    

#################################################################
# MODEL LEARNING FUNCTION
#################################################################
def learnModel(m, kernelw, demand, c0, constants):

    T, Fj, aij, ci = constants

    #values for target function
    u = [[0.0 for y in range(3)] for x in range(m)] #u[k,j]
    b = [0.0 for x in range(3)] #b[j]

    # Create a new model
    qp_model = gb.Model('qp')

    # Create alpha variables: index = ((k-1) * T + (t-1))*3 + (j-1)
    alpha = qp_model.addVars(m * T * 3, lb=0.0, vtype=gb.GRB.CONTINUOUS, name="alpha") 

    # Create beta variables:  index = ((k-1) * T + (t-1))*3 + (j-1)
    beta = qp_model.addVars(m * T * 3, lb=0.0, vtype=gb.GRB.CONTINUOUS, name="beta") 

    # Create epsilon variables: index = (k-1) * 3 + (j-1)
    epsilon = qp_model.addVars(m * 3, lb=0.0, vtype=gb.GRB.CONTINUOUS, name="epsilon") 


    # Set objective
    obj = - 1.0/(4.0)*gb.quicksum( #p
                            gb.quicksum( #q
                                (
                                    gb.quicksum( #j=1..3
                                        ( 1.0/c0[j-1] *
                                            (gb.quicksum(beta[((p-1) * T + (t-1))*3 + (j-1)] for t in range(1,T+1)) + epsilon[(p-1) * 3 + (j-1)] - Fj[j-1] ) *
                                            (gb.quicksum(beta[((q-1) * T + (t-1))*3 + (j-1)] for t in range(1,T+1)) + epsilon[(q-1) * 3 + (j-1)] - Fj[j-1] )
                                        )
                                        for j in range(1,3+1))
                                ) * kernelw[p - 1][q - 1]
                                for q in range(1,m+1) )
                            for p in range(1,m+1) ) +\
          gb.quicksum(
            gb.quicksum(
                gb.quicksum(
                    ((ci[i-1] - alpha[((k-1) * T + (t-1))*3 + (i-1)]) * demand[k-1][t-1][i-1])
                    for i in range(1,3+1))
                for t in range(1,T+1))
            for k in range(1,m+1) )

    qp_model.setObjective(obj, sense=gb.GRB.MAXIMIZE)

    #add constraints alpha+beta >= aij
    for k in range(1,m+1):
        for t in range(1,T+1):
            for j in range(1,3+1):
                qp_model.addConstrs(
                    ((alpha[((k-1) * T + (t-1))*3 + (i-1)] + beta[((k-1) * T + (t-1))*3 + (j-1)]) >= aij[i-1][j-1]
                     for i in range(j,3+1))
                    )


    #add constraints sum_ujk = 0
    qp_model.addConstrs(
        (gb.quicksum(
            (gb.quicksum(beta[((k-1) * T + (t-1))*3 + (j-1)] for t in range(1,T+1)) + epsilon[(k-1) * 3 + (j-1)] - Fj[j-1] )
            for k in range(1,m+1)) == 0)
        for j in range(1,3+1)
        )

    qp_model.optimize()


    # calculate u[k][j]
    for k in range(1,m+1):
        for j in range(1,3+1):
            beta_sum_tmp=0
            for t in range(1,T+1):
                beta_sum_tmp = beta_sum_tmp + beta[((k-1) * T + (t-1))*3 + (j-1)].x;
            u[k-1][j-1] = beta_sum_tmp + epsilon[(k-1) * 3 + (j-1)].x - Fj[j-1];

    #optimize for b

    # Create a new model
    lp_model = gb.Model('mylp')

    # Create b variables: index
    bvar = lp_model.addVars(3, lb=-gb.GRB.INFINITY, vtype=gb.GRB.CONTINUOUS, name="bvars") 

    # create y variables: index = (((k-1) * T + (t-1))*3 + (j-1)) * 3 + (i-1)
    yijkt = lp_model.addVars(3 * 3 * m * T, lb=0.0, vtype=gb.GRB.CONTINUOUS, name="y_var") 

    objLP = gb.quicksum( #k
            (gb.quicksum( #j
                Fj[j-1] *
                (1/(2) * gb.quicksum( 1.0/c0[j-1] *
                        u[p-1][j-1] * kernelw[p - 1][k - 1]
                        for p in range(1,m+1)) - bvar[j-1])
                for j in range(1,3+1))
             - gb.quicksum( #t
                (gb.quicksum( #j
                    gb.quicksum( #i
                        (aij[i-1][j-1] * yijkt[(((k-1) * T + (t-1))*3 + (j-1)) * 3 + (i-1)])
                        for i in range(1,3+1))
                    for j in range(1,3+1))
                 - gb.quicksum( #i
                    ci[i-1] * demand[k-1][t-1][i-1]
                    for i in range(1,3+1))
                )
                for t in range(1,T+1))
            )
            for k in range(1,m+1))

    lp_model.setObjective(objLP, sense=gb.GRB.MINIMIZE)

    #constraints on assigning to demand
    for i in range(1,3+1):
        for t in range(1,T+1):
            lp_model.addConstrs(
                (gb.quicksum( #j
                    yijkt[(((k-1) * T + (t-1))*3 + (j-1)) * 3 + (i-1)]
                    for j in range(1,3+1))
                    <= demand[k-1][t-1][i-1]
                for k in range(1,m+1))
                )

    #constraints on assigning resources
    for j in range(1,3+1):
        for t in range(1,T+1):
            lp_model.addConstrs(
                (gb.quicksum( #i
                    yijkt[(((k-1) * T + (t-1))*3 + (j-1)) * 3 + (i-1)]
                    for i in range(1,3+1))
                    <= (1/(2*c0[j-1]) * gb.quicksum( #for j=1, p
                        u[p-1][j-1] * kernelw[p - 1][k - 1]
                        for p in range(1,m+1)) - bvar[j-1])
                for k in range(1,m+1))
                )

    #constraint for i<j: y=0
    for k in range(1,m+1):
        for i in range(1,3+1):
            for j in range(1,3+1):
                if(i<j):
                    lp_model.addConstrs(
                        yijkt[(((k-1) * T + (t-1))*3 + (j-1)) * 3 + (i-1)]==0
                        for t in range(1,T+1))

    #constraints on positive capacity
    for j in range(1,3+1):
        lp_model.addConstrs(
            (1.0 / (2*c0[j-1]) * gb.quicksum(u[p-1][j-1] * kernelw[p - 1][k - 1] for p in range(1, m + 1)) - bvar[j - 1]) >= 0
            for k in range(1,m+1))

    lp_model.optimize()

    # calculate b[j]
    for j in range(1,3+1):
        b[j-1] = bvar[j-1].x


    return u, b


#################################################################
# PRESCRIPTION FUNCTION
#################################################################
def prescribe(u, b, m, m_pred, kernelw, c0):

    #prescriptions
    q = [[0.0 for y in range(3)] for x in range(m_pred)] #q[k,j]

    for l in range(m_pred):
        for j in range(3):
            q[l][j] = 0
            for k in range(m):
                q[l][j] = q[l][j] + u[k][j] * kernelw[k][l]
            q[l][j] = q[l][j] * 1.0 / (2.0 * c0[j]) - b[j]

    return q

#################################################################
# PROFIT EVALUATION FUNCTION
#################################################################
def evaluateProfit(m_pred, q, demand, constants):

    T, Fj, aij, ci = constants
    totprofit = 0.0

    #Demand[k][t][i]

    for l in range(m_pred): 

        for j in range(3):
            totprofit = totprofit - Fj[j]*q[l][j];

        for t in range(T): 

            for i in range(3):
                totprofit = totprofit - ci[i]*demand[l][t][i];

            availRes = q[l][:]; 
            remDem = demand[l][t][:]; 
            renderService = [[0.0 for y in range(3)] for x in range(3)]; #renderService[i][j]
            for i in range(3): 
                renderService[i][i] = min(availRes[i], remDem[i]);
                availRes[i] = availRes[i] - renderService[i][i];
                remDem[i] = remDem[i] - renderService[i][i];

            for j in range(2): 
                if aij[j+1][j] > 0:
                    renderService[j+1][j]= min(availRes[j], remDem[j+1]);
                    availRes[j] = availRes[j] - renderService[j+1][j];
                    remDem[j+1] = remDem[j+1] - renderService[j+1][j];

            if aij[2][0] > 0:
                renderService[2][0]= min(availRes[0], remDem[2]);
                availRes[0] = availRes[0] - renderService[2][0];
                remDem[2] =  remDem[2] - renderService[2][0];

            for i in range(3):
                for j in range(3):
                    totprofit = totprofit + aij[i][j] * renderService[i][j];

    return totprofit

