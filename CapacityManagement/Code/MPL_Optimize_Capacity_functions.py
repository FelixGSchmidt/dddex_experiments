
# Optimization using SAA or ExPost-based approaches


import gurobipy as gb
import csv
import numpy.random


#################################################################
# Optimize Capacity Deterministic FUNCTION
#################################################################
def optimizeDetermCap(inputFile, outputFile, numYears, configFile):

    demandSL = []
    demandMP = []
    demandHE = []
    dateVals = []
    yearVals = []
    weekVals = []
    weekDayVals = []

    maxNumConfigs = 1000;


    dataArrayLen = 0
    with open(inputFile) as csvfile:
        myreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in myreader:
            if dataArrayLen>0: #skip first line: column names
                demandSL.append(float(row[5].replace(',', '.')))
                demandMP.append(float(row[6].replace(',', '.')))
                demandHE.append(float(row[7].replace(',', '.')))
                dateVals.append(row[0])
                yearVals.append(int(row[1]))
                weekVals.append(int(row[2]))
                weekDayVals.append(int(row[3]))
            dataArrayLen += 1

    #correct for data array len (skipped first line)
    dataArrayLen -= 1


    demandSLByYearWeekAligned = [[[0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    demandMPByYearWeekAligned = [[[0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    demandHEByYearWeekAligned = [[[0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    daysPerWeekByYearWeek = [[0 for y in range(53)] for x in range(numYears)] #daysPerWeekByYearWeek[yearID][weekID]

    capReservationValsByYearWeek = [[[[0 for z in range(3)] for y in range(53)] for x in range(numYears)] for conf in range (maxNumConfigs)] #baseValsByYearWeek[configID][yearID][weekID][N]
    flexValsByYearWeek = [[[0 for z in range(5)] for y in range(53)] for x in range(numYears)] #flexValsByYearWeek[yearID][weekID][weekdayID]
    shiftValsByYearWeek = [[[0 for z in range(5)] for y in range(53)] for x in range(numYears)] #shiftValsByYearWeek[yearID][weekID][weekdayID]

    #resort data into new structures
    for j in range(dataArrayLen):
        demandSLByYearWeekAligned[yearVals[j]-2014][weekVals[j]-1][ daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1] ]=demandSL[j]; #values left aligned in array, independent of which days in week have demand
        demandMPByYearWeekAligned[yearVals[j]-2014][weekVals[j]-1][ daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1] ]=demandMP[j]; #values left aligned in array, independent of which days in week have demand
        demandHEByYearWeekAligned[yearVals[j]-2014][weekVals[j]-1][ daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1] ]=demandHE[j]; #values left aligned in array, independent of which days in week have demand
        daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1]=daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1]+1;


    #const

    aij = [[0.0 for x in range(3*3)] for y in range(maxNumConfigs)];
    dit = [0.0 for x in range(3*5)];
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

    #set revenue for resource j providing service i
    for k in range(numConfigLines):
        for i in range(3):
            for j in range(3):
                aij[k][3*i + j] =pi[k][i] - vj[k][j] + ci[k][i];

    # for each week and all years -> optimize on week basis
    for yearID in range(numYears):
        for weekID in range(53):

            T = daysPerWeekByYearWeek[yearID][weekID]

            if(T==0):
                continue

            #prepare demand values of week
            for t in range(T):
                dit[t*3 + 0] = demandSLByYearWeekAligned[yearID][weekID][t];
                dit[t*3 + 1] = demandMPByYearWeekAligned[yearID][weekID][t];
                dit[t*3 + 2] = demandHEByYearWeekAligned[yearID][weekID][t];

            optModel = gb.Model("continuous1")

            qj = optModel.addVars(3, lb=0, ub=1000, vtype=gb.GRB.CONTINUOUS, name="q_j") #3 variables for capacity reservation

            yijt = optModel.addVars(3*3*T, lb=0, ub=1000, vtype=gb.GRB.CONTINUOUS, name="y_ijt") #one base capacity variable


            #cannot over deliver on demand
            for i in range(3):
                optModel.addConstrs(
                    ( gb.quicksum( yijt[9*t + 3*i + j] for j in range(3)) <= dit[t*3 + i]
                        for t in range(T))
                )

            #cannot use more resources than reserved
            for j in range(3):
                optModel.addConstrs(
                    ( gb.quicksum( yijt[9*t + 3*i + j] for i in range(3)) <= qj[j]
                        for t in range(T))
                )

            #assignment not possible for lower-qualified resources to higher-demanding services
            for i in range(3):
                for j in range(3):
                    if(i<j):
                        optModel.addConstrs((yijt[9*t + 3*i + j] == 0 for t in range(T)) )


            # configuration parameter dependent: optimize for all parameter configurations

            for configID in range(numConfigLines):
                #set optimization objective: cost function
                optModel.setObjective(gb.quicksum(gb.quicksum(
                                                        gb.quicksum( (aij[configID][3*i + j] * yijt[9*t + 3*i + j])
                                                            for j in range(3)) -
                                                        (ci[configID][i] * dit[t*3 + i])
                                                  for i in range(3))
                                              for t in range(T)) -
                                      gb.quicksum(fj[configID][j] * qj[j]
                                          for j in range(3))
                                      , gb.GRB.MAXIMIZE)

                optModel.optimize()

                for i in range(3):
                    capReservationValsByYearWeek[configID][yearID][weekID][i]=qj[i].x


    #write results in csv
    with open(outputFile, 'w') as csvfile: #newline=''
        mywriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        mywriter.writerow(["ConfigID", "Date", "CapReservation1", "CapReservation2", "CapReservation3"])
        for configID in range(numConfigLines):
            for j in range(dataArrayLen):
                mywriter.writerow([configID, dateVals[j],
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][yearVals[j] - 2014][weekVals[j] - 1][0] )).replace(".",","),
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][yearVals[j] - 2014][weekVals[j] - 1][1] )).replace(".",","),
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][yearVals[j] - 2014][weekVals[j] - 1][2] )).replace(".",",")])


    return 0;


#################################################################
# Optimize Capacity Sample Average Approximation FUNCTION
#################################################################
def optimizeSAA_Cap(inputFile, outputFile, numYears, configFile, trainSize):
    # read csv
    demandSL = []
    demandMP = []
    demandHE = []
    dateVals = []
    yearVals = []
    weekVals = []
    weekDayVals = []

    maxNumConfigs = 1000;


    nsamples = 53 * 3; ## number of weeks in 3 years - only for array size


    dataArrayLen = 0
    with open(inputFile) as csvfile:
        myreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in myreader:
            #print(', '.join(row))
            if dataArrayLen>0: #skip first line: column names
                #demandVals.append(int(row[4]))
                demandSL.append(float(row[5].replace(',', '.')))
                demandMP.append(float(row[6].replace(',', '.')))
                demandHE.append(float(row[7].replace(',', '.')))
                dateVals.append(row[0])
                yearVals.append(int(row[1]))
                weekVals.append(int(row[2]))
                weekDayVals.append(int(row[3]))
            dataArrayLen += 1

    #correct for data array len (skipped first line)
    dataArrayLen -= 1



    demandSLByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    demandMPByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    demandHEByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    daysPerWeekByYearWeek = [[0 for y in range(53)] for x in range(numYears)] #daysPerWeekByYearWeek[yearID][weekID]

    capReservationValsByYearWeek = [[[[0.0 for z in range(3)] for y in range(53)] for x in range(numYears)] for conf in range (maxNumConfigs)] #baseValsByYearWeek[configID][yearID][weekID][N]
    flexValsByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #flexValsByYearWeek[yearID][weekID][weekdayID]
    shiftValsByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #shiftValsByYearWeek[yearID][weekID][weekdayID]

    #resort data into new structures
    for j in range(dataArrayLen):
        daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1]=daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1]+1;
        demandSLByYearWeek[yearVals[j]-2014][weekVals[j]-1][weekDayVals[j]-1]=demandSL[j];
        demandMPByYearWeek[yearVals[j]-2014][weekVals[j]-1][weekDayVals[j]-1]=demandMP[j];
        demandHEByYearWeek[yearVals[j]-2014][weekVals[j]-1][weekDayVals[j]-1]=demandHE[j];

    #const
    aij = [[0.0 for x in range(3*3)] for y in range(maxNumConfigs)];
    dit = [[0.0 for y in range(nsamples)] for x in range(3*5)]; #nsamples for each day and i: dit[t/i][sampleID_k]
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



    #set revenue for resource j providing service i
    for k in range(numConfigLines):
        for i in range(3):
            for j in range(3):
                aij[k][3*i + j] =pi[k][i] - vj[k][j] + ci[k][i];

    j=0;

    for yearID in range(numYears-1): ## for all training years
        for weekID in range(53):
            ## check if week exists
            if(0==daysPerWeekByYearWeek[yearID][weekID]):
                continue

            for t in range(5):
                dit[t*3 + 0][j] = demandSLByYearWeek[yearID][weekID][t];
                dit[t*3 + 1][j] = demandMPByYearWeek[yearID][weekID][t];
                dit[t*3 + 2][j] = demandHEByYearWeek[yearID][weekID][t];

            j=j+1;

    nsamples_max = j; ## have nsamples of one week each

    nsamples = trainSize

    nsamples_min = nsamples_max - nsamples

    print("Using only ", nsamples, " samples..")



    T=5
    optModel = gb.Model("continuous1")

    qj = optModel.addVars(3, lb=0, ub=10000, vtype=gb.GRB.CONTINUOUS, name="q_j") #T variables for flexible capacity

    yijt = optModel.addVars(3*3*T * nsamples, lb=0, ub=10000, vtype=gb.GRB.CONTINUOUS, name="y_ijt") #one base capacity variable
    # yijt[((t*3+i)*3 + j) * nsamples + k]

    #cannot over deliver on demand
    for k in range(nsamples):
        for i in range(3):
            optModel.addConstrs(
                ( gb.quicksum( yijt[((t*3+i)*3 + j) * nsamples + k] for j in range(3)) <= dit[t*3 + i][nsamples_min+k]
                    for t in range(T))
            )

    #cannot use more resources than reserved
    for k in range(nsamples):
        for j in range(3):
            optModel.addConstrs(
                ( gb.quicksum( yijt[((t*3+i)*3 + j) * nsamples + k] for i in range(3)) <= qj[j]
                    for t in range(T))
            )

    #assignment not possible for lower-qualified resources to higher-demanding services
    for t in range(T):
        for i in range(3):
            for j in range(3):
                if(i<j):
                    optModel.addConstrs((yijt[((t*3+i)*3 + j) * nsamples + k] == 0 for k in range(nsamples)) )

    for configID in range(numConfigLines):
        #set optimization objective: cost function
        optModel.setObjective(gb.quicksum(
                                            gb.quicksum(gb.quicksum(
                                                gb.quicksum( aij[configID][3*i + j] * yijt[((t*3+i)*3 + j) * nsamples + k]
                                                    for j in range(3)) -
                                                ci[configID][i] * dit[t*3 + i][nsamples_min+k]
                                          for i in range(3))
                                      for t in range(T)) -
                              gb.quicksum(fj[configID][j] * qj[j]
                                  for j in range(3))
                             for k in range(nsamples))
                              , gb.GRB.MAXIMIZE)
        optModel.optimize()

        for i in range(3):
            capReservationValsByYearWeek[configID][0][0][i]=qj[i].x


    #write results in csv
    with open(outputFile, 'w') as csvfile: #, newline=''
        mywriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        mywriter.writerow(["ConfigID", "Date", "CapReservation1", "CapReservation2", "CapReservation3"])
        for configID in range(numConfigLines):
            for j in range(dataArrayLen):
                mywriter.writerow([configID, dateVals[j],
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][0][0][0] )).replace(".",","),
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][0][0][1] )).replace(".",","),
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][0][0][2] )).replace(".",",")])

    return 0;


#################################################################
# Optimize Capacity Kernel Sample Average Approximation FUNCTION
#################################################################
def optimizeKernelSAA_Cap(inputFile, outputFile, numYears, configFile, trainSize, kernelFilename):

    ##load kernel
    m1 = 157
    m2 = 52
    KernelW = [[0.0 for y in range(m2)] for x in range(m1)] #KernelW[157][52]

    with open(kernelFilename) as csvfile:
            myreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            i=-1;
            for row in myreader:
                if i >= 0: ## skip first line
                    for j in range(m2):
                        KernelW[i][j] = float(row[j].replace(',','.'))
                i = i+1;


    # read csv
    demandSL = []
    demandMP = []
    demandHE = []
    dateVals = []
    yearVals = []
    weekVals = []
    weekDayVals = []

    maxNumConfigs = 1000;


    nsamples = 53 * 3; ## number of weeks in 3 years


    dataArrayLen = 0
    with open(inputFile) as csvfile:
        myreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in myreader:
            if dataArrayLen>0: #skip first line: column names
                demandSL.append(float(row[5].replace(',', '.')))
                demandMP.append(float(row[6].replace(',', '.')))
                demandHE.append(float(row[7].replace(',', '.')))
                dateVals.append(row[0])
                yearVals.append(int(row[1]))
                weekVals.append(int(row[2]))
                weekDayVals.append(int(row[3]))
            dataArrayLen += 1

    #correct for data array len (skipped first line)
    dataArrayLen -= 1


    demandSLByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    demandMPByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    demandHEByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #demandByYearWeek[yearID][weekID][weekdayID]
    daysPerWeekByYearWeek = [[0 for y in range(53)] for x in range(numYears)] #daysPerWeekByYearWeek[yearID][weekID]

    capReservationValsByYearWeek = [[[[0.0 for z in range(3)] for y in range(53)] for x in range(numYears)] for conf in range (maxNumConfigs)] #baseValsByYearWeek[configID][yearID][weekID][N]
    flexValsByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #flexValsByYearWeek[yearID][weekID][weekdayID]
    shiftValsByYearWeek = [[[0.0 for z in range(5)] for y in range(53)] for x in range(numYears)] #shiftValsByYearWeek[yearID][weekID][weekdayID]

    #resort data into new structures
    for j in range(dataArrayLen):
        daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1]=daysPerWeekByYearWeek[yearVals[j]-2014][weekVals[j]-1]+1;
        demandSLByYearWeek[yearVals[j]-2014][weekVals[j]-1][weekDayVals[j]-1]=demandSL[j];
        demandMPByYearWeek[yearVals[j]-2014][weekVals[j]-1][weekDayVals[j]-1]=demandMP[j];
        demandHEByYearWeek[yearVals[j]-2014][weekVals[j]-1][weekDayVals[j]-1]=demandHE[j];

    #const
    aij = [[0.0 for x in range(3*3)] for y in range(maxNumConfigs)];
    dit = [[0.0 for y in range(nsamples)] for x in range(3*5)]; #nsamples for each day and i: dit[t/i][sampleID_k]
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


    #set revenue for resource j providing service i
    for k in range(numConfigLines):
        for i in range(3):
            for j in range(3):
                aij[k][3*i + j] =pi[k][i] - vj[k][j] + ci[k][i];

    j=0;

    for yearID in range(numYears-1): ## for all training years
        for weekID in range(53):
            ## check if week exists
            if(0==daysPerWeekByYearWeek[yearID][weekID]):
                continue

            for t in range(5):
                dit[t*3 + 0][j] = demandSLByYearWeek[yearID][weekID][t];
                dit[t*3 + 1][j] = demandMPByYearWeek[yearID][weekID][t];
                dit[t*3 + 2][j] = demandHEByYearWeek[yearID][weekID][t];

            j=j+1;

    nsamples_max = j; ## have nsamples of one week each

    nsamples = trainSize

    nsamples_min = nsamples_max - nsamples

    print("Using only ", nsamples, " samples..")


    T=5
    optModel = gb.Model("continuous1")

    qj = optModel.addVars(3, lb=0, ub=10000, vtype=gb.GRB.CONTINUOUS, name="q_j") #T variables for flexible capacity

    yijt = optModel.addVars(3*3*T * nsamples, lb=0, ub=10000, vtype=gb.GRB.CONTINUOUS, name="y_ijt") #one base capacity variable
    # yijt[((t*3+i)*3 + j) * nsamples + k]

    #cannot over deliver on demand
    for k in range(nsamples):
        for i in range(3):
            optModel.addConstrs(
                ( gb.quicksum( yijt[((t*3+i)*3 + j) * nsamples + k] for j in range(3)) <= dit[t*3 + i][nsamples_min+k]
                    for t in range(T))
            )

    #cannot use more resources than reserved
    for k in range(nsamples):
        for j in range(3):
            optModel.addConstrs(
                ( gb.quicksum( yijt[((t*3+i)*3 + j) * nsamples + k] for i in range(3)) <= qj[j]
                    for t in range(T))
            )

    #assignment not possible for lower-qualified resources to higher-demanding services
    for t in range(T):
        for i in range(3):
            for j in range(3):
                if(i<j):
                    optModel.addConstrs((yijt[((t*3+i)*3 + j) * nsamples + k] == 0 for k in range(nsamples)) )

    for configID in range(numConfigLines):

        for weekID in range(52): ## for all 52 test weeks
            #set optimization objective: cost function
            optModel.setObjective(gb.quicksum(KernelW[k][weekID] *
                                              (gb.quicksum(gb.quicksum(
                                                    gb.quicksum( aij[configID][3*i + j] * yijt[((t*3+i)*3 + j) * nsamples + k]
                                                        for j in range(3)) -
                                                    ci[configID][i] * dit[t*3 + i][nsamples_min+k]
                                              for i in range(3))
                                          for t in range(T)) -
                                  gb.quicksum(fj[configID][j] * qj[j]
                                      for j in range(3)))
                                 for k in range(nsamples))
                                  , gb.GRB.MAXIMIZE)
            optModel.optimize()

            for i in range(3):
                capReservationValsByYearWeek[configID][0][weekID][i]=qj[i].x

    #write results in csv
    with open(outputFile, 'w') as csvfile: #, newline=''
        mywriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        mywriter.writerow(["ConfigID", "WeekID", "CapReservation1", "CapReservation2", "CapReservation3"])
        for configID in range(numConfigLines):
            for weekID in range(52):
                mywriter.writerow([configID, weekID+1,
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][0][weekID][0] )).replace(".",","),
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][0][weekID][1] )).replace(".",","),
                                   ("{:.12f}".format( capReservationValsByYearWeek[configID][0][weekID][2] )).replace(".",",")])

    return 0;
