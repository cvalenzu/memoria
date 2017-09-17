import numpy as np
from numba import jit

@jit
def getDataWindowed(data,inSize,outSize):
    biggest = np.max([inSize,outSize])

    matrixIn = np.zeros((len(data)-2*biggest, inSize))
    matrixOut = np.zeros((len(data)-2*biggest, outSize))
    for i in range(len(data)-2*biggest):
        matrixIn[i,:] = data[i:i+inSize]
        matrixOut[i,:] = data[i+inSize+1:i+inSize+outSize+1]
    return matrixIn,matrixOut


#Metrics
def mape_score(y_test,y_pred):
    error = np.abs((y_test - y_pred)/np.mean(y_test))
    return np.average(error)

def nmse_score(y_test,y_pred):
    error = (y_test-y_pred)**2 / (np.mean(y_test)* np.mean(y_pred))
    return np.average(error)
