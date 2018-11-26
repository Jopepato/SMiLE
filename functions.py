#!/usr/bin/python3
import sys
import numpy as np
import scipy.sparse as sp
import random
import math
import gc

def readDataFromFile (fileName):
    "This functions reads data from a file and store it in two matrices"
    #Open the file
    file = open(fileName, 'r')

    #Now we have to read the first line and check if it's sparse or dense
    firstLine = file.readline()
    words = firstLine.split()
    word = words[1]
    if word[:-1] == 'SPARSE':
        sparse = True #The file is in sparse mode
    else:
        sparse = False #The file is in dense mode


    secondLine = file.readline()
    words = secondLine.split()
    instances = int(words[1])
    thirdLine = file.readline()
    words = thirdLine.split()
    attributes = int(words[1])
    fourthLine = file.readline()
    words = fourthLine.split()
    labels = int(words[1])
    #Now we do a loop reading all the other lines
    #Then we read the file, different way depending if sparse or dense

    #The loop starts in the first line of data
    #We have to store that data in two matrices
    X = np.zeros((instances, attributes), dtype=float)
    y = np.zeros((instances, labels), dtype=int)
    numberLine = 0
    for line in file.readlines():
        putToX = True
        firstIndex = 1
        numberData = 0
        numberY = 0
        for data in line.split():
            if sparse:#Sparse format, we have to split each data 
                if data == '[':
                    putToX = False

                if putToX == True and (data != '[' and data != ']'):
                    sparseArray = data.split(':')
                    lastIndex = int(sparseArray[0])
                    for i in range(firstIndex, lastIndex - 1):
                        X[numberLine, i-1] = float(0)
                    X[numberLine, lastIndex-1] = float(sparseArray[1])
                    firstIndex = lastIndex-1
                else:
                    if (data != '[') and (data != ']'):
                        aux = float(data)
                        y[numberLine, numberY] = int(aux)
                        numberY += 1
                
            else:#Dense format
                if data == '[':
                    putToX = False

                if putToX == True and (data != '[' and data != ']'):
                    X[numberLine, numberData] = float(data)
                else:
                    if (data != '[') and (data != ']'):
                        #This is good for the dense format
                        aux = float(data)
                        y[numberLine, numberY] = int(aux)
                        numberY += 1
            numberData += 1
        
        numberLine += 1
    X = sp.csr_matrix(X)
    file.close()
    return X, y

def readParams (paramsFileName):
    "This functions reads the params from a file and store them in a dictionary"
    file = open(paramsFileName, 'r')
    
    paramNames = []
    paramAttributes = []
    #We are going to create a dictionary from both lists
    for line in file.readlines():
        data = line.split(' : ')
        paramNames.append(str(data[0]))
        aux = data[1]
        paramAttributes.append(aux[0:len(aux)-1])

    paramDictionary = {}

    file.close()

    for i in range(len(paramNames)):
        
        paramDictionary[paramNames[i]] = paramAttributes[i]

    del paramNames
    del paramAttributes
    return paramDictionary

def storeResults(paramsUsed, resultsObtained, nameResultFile):
    "This function stores the results obtained into a file"
    file = open(nameResultFile, 'w')
    #Now we will write the params used in the experiment to the file
    for i in paramsUsed.items():
        file.write(str(i[0]) + ' : ' + str(i[1]) + '\n')
    #Now we will write the results obtained in the experiment to the file
    for i in resultsObtained:
        file.write('@' + str(i[0]) + ' : ' + str(i[1])+ '\n')

    file.write('@Ok\n')
    file.close()

def labelMetrics(y_test):
    "This function will return the label metrics as Label Density and Label Cardinality"
    labelCardinality = 0.0
    labelDensity = 0.0

    for i in range(y_test.shape[0]):
        aux = 0.0
        for j in range(y_test.shape[1]):
            labelCardinality = labelCardinality + int(y_test[i,j])
            aux = aux + int(y_test[i,j])
        
        labelDensity = labelDensity + (aux/y_test.shape[1])
    
    labelCardinality = labelCardinality/y_test.shape[0]
    labelDensity = labelDensity/y_test.shape[0]

    return labelCardinality, labelDensity


def exampleBasedMetrics(y_test, predictions, beta=1):
    "This functions returns the different exampleBased metrics for our predictions"
    #We first calculate the subsetAccuracy
    subsetAccuracy = 0.0
    for i in range(y_test.shape[0]):
        same = True
        for j in range(y_test.shape[1]):
            if y_test[i,j] != predictions[i,j]:
                same = False
                break
        if same:
            subsetAccuracy = subsetAccuracy + 1

    subsetAccuracy = subsetAccuracy/y_test.shape[0]

    #Then we can calculate the haming loss
    hloss = 0.0
    for i in range(y_test.shape[0]):
        aux = 0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) != int(predictions[i,j]):
                aux = aux+1
        aux = aux/y_test.shape[1]
        hloss = hloss + aux

    hloss = float(hloss/y_test.shape[0])

    #And then all the others parameters
    accuracy = 0.0
    precision = 0.0
    recall = 0.0

    for i in range(y_test.shape[0]):
        #We have to calculate the intersections, the union and the entire vector modules
        intersection = 0.0
        union = 0.0
        hXi = 0.0
        Yi = 0.0
        for j in range(y_test.shape[1]):
            Yi = Yi + int(y_test[i,j])
            hXi = hXi + int(predictions[i,j])
            if int(y_test[i,j]) == 1 or int(predictions[i,j]) == 1:
                union = union + 1
            if y_test[i,j] == 1 and int(predictions[i,j]) == 1:
                intersection = intersection + 1
    
        if union != 0:
            accuracy = accuracy + float(intersection/union)
        if hXi != 0:
            precision = precision + float(intersection/hXi)
        if Yi != 0:
            recall = recall + float(intersection/Yi)
    
    accuracy = float(accuracy/y_test.shape[0])
    precision = float(precision/y_test.shape[0])
    recall = float(recall/y_test.shape[0])
    FBeta = 0.0
    num = float((1+pow(beta,2))*precision*recall)
    den = float(pow(beta,2)*precision + recall)
    if den != 0:
        FBeta = num/den

    return subsetAccuracy, hloss, accuracy, precision, recall, FBeta

def exampleBasedRankingMetrics(y_test, ranking):
    "This function returns the differen ranked metrics"

    oneError = 0.0
    coverage = 0.0
    rankingLoss = 0.0
    averagePrecision = 0.0
    for i in range(y_test.shape[0]):
        if y_test[i, ranking[i,0]-1] != 1:
            oneError += 1
        relevantVector = []
        for j in range(y_test.shape[1]):
            #We construct a vector adding the relevant indexes
            if y_test[i,j] == 1:
                relevantVector.append(j+1)
        #Now that we have created the vector of the important indexes we can check some metrics
        count = len(relevantVector)
        for j in range(ranking.shape[1]):
            if ranking[i,j] in relevantVector:
                count = count - 1
            if count == 0:
                coverage = coverage + j
                break
        numFound = 0
        #Now we are going to do the ranking loss
        for j in range(ranking.shape[1]):
            if ranking[i,j] in relevantVector:
                averagePrecision = averagePrecision + (j-numFound)
                numFound = numFound +1
            if numFound == len(relevantVector):
                break

    oneError = oneError/y_test.shape[0]
    coverage = coverage/y_test.shape[0]
    averagePrecision = averagePrecision/y_test.shape[0]

    return oneError, coverage, rankingLoss, averagePrecision

def labelBasedMetrics(y_test, predictions, beta = 1):
    "This function returns the different labelBased metrics for our results"
    TP = []
    FP = []
    TN = []
    FN = []
    for j in range(y_test.shape[1]):
        TPaux = 0
        FPaux = 0
        TNaux = 0
        FNaux = 0
        for i in range(y_test.shape[0]):
            if int(y_test[i,j]) == 1:
                if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                    TPaux += 1
                else:
                    FPaux += 1

            else:
                if int(y_test[i,j]) == 0 and int(predictions[i,j]) == 0:
                    TNaux += 1
                else:
                    FNaux += 1
        TP.append(TPaux)
        FP.append(FPaux)
        TN.append(TNaux)
        FN.append(FNaux)
    
    #Now that we have the different metrics for every label we can get the diferent ones
    #Now we will do the micro and macro averaging for Accuracy, Precision, Recall and FBeta
    AccuracyMacro = 0.0
    PrecisionMacro = 0.0
    RecallMacro = 0.0
    FBetaMacro = 0.0
    AccuracyMicro = 0.0
    PrecisionMicro = 0.0
    RecallMicro = 0.0
    FBetaMicro = 0.0

    TPMicro = 0.0
    FPMicro = 0.0
    TNMicro = 0.0
    FNMicro = 0.0
    for j in range(0, len(TP)):
        TPMicro = TPMicro + TP[j]
        FPMicro = FPMicro + FP[j]
        TNMicro = TNMicro + TN[j]
        FNMicro = FNMicro + FN[j]
        AccuracyMacro = AccuracyMacro + ((TP[j] + TN[j])/(TP[j] + FP[j] + TN[j] + FN[j]))
        if TP[j] + FP[j] != 0:
            PrecisionMacro = PrecisionMacro + (TP[j]/(TP[j] + FP[j]))
        if TP[j] + FN[j] != 0:
            RecallMacro = RecallMacro + (TP[j]/(TP[j] + FN[j]))
        num = float((1+pow(beta,2))*TP[j])
        den = float((1+pow(beta,2))*TP[j] + pow(beta,2)*FN[j] + FP[j])
        if den != 0:
            FBetaMacro = FBetaMacro + num/den

    AccuracyMacro = float(AccuracyMacro / len(TP))
    PrecisionMacro = float(PrecisionMacro / len(TP))
    RecallMacro = float(RecallMacro / len(TP))
    FBetaMacro = float(FBetaMacro / len(TP))
    if (TPMicro + FPMicro + TNMicro + FNMicro) != 0:
        AccuracyMicro = float((TPMicro + TNMicro)/(TPMicro + FPMicro + TNMicro + FNMicro))
    if (TPMicro + FPMicro) != 0:
        PrecisionMicro = float(TPMicro/(TPMicro + FPMicro))
    if (TPMicro + FNMicro) != 0:
        RecallMicro = float(TPMicro/(TPMicro + FNMicro))
    num = float((1+pow(beta,2))*TPMicro)
    den = float((1+pow(beta,2))*TPMicro + pow(beta,2)*FNMicro + FPMicro)
    if den != 0:
        FBetaMicro = num/den

    return AccuracyMacro, PrecisionMacro, RecallMacro, FBetaMacro, AccuracyMicro, PrecisionMicro, RecallMicro, FBetaMicro

def getAccuracy(y_test, predictions):
    accuracy = 0.0

    for i in range(y_test.shape[0]):
        #We have to calculate the intersections, the union and the entire vector modules
        intersection = 0.0
        union = 0.0
        hXi = 0.0
        Yi = 0.0
        for j in range(y_test.shape[1]):
            Yi = Yi + int(y_test[i,j])
            hXi = hXi + int(predictions[i,j])
            if int(y_test[i,j]) == 1 or int(predictions[i,j]) == 1:
                union = union + 1
            if y_test[i,j] == 1 and int(predictions[i,j]) == 1:
                intersection = intersection + 1
    
        if union != 0:
            accuracy = accuracy + float(intersection/union)
    
    accuracy = float(accuracy/y_test.shape[0])


    return accuracy