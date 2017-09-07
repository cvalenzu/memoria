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
