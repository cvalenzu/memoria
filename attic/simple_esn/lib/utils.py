import numpy as np

def getDataWindowed(data,inSize,outSize):
    
    matrixIn = np.zeros((len(data)-outSize-inSize+1, inSize))
    matrixOut = np.zeros((len(data)-outSize-inSize+1, outSize))
    
    for i in range(len(data)-inSize-outSize+1):
        matrixIn[i,:] = data[i:i+inSize]
        matrixOut[i,:] = data[i+inSize:i+inSize+outSize]
    return matrixIn,matrixOut


def createFolds(dataSize, k):
    vector = np.arange(dataSize)
    splitted = np.array_split(vector,k+1)
    
    folds = []
    
    test_set = []
    for i in range(k):
        test_set = np.hstack((test_set, splitted[i]))
        val_set = splitted[i+1]
        folds.append((test_set.astype('int'),val_set.astype('int')))
    return folds

if __name__ == "__main__":
    data = np.loadtxt('MackeyGlass_t17.txt')

    input_steps = 12
    prediction_steps = 12

    X,y = getDataWindowed(data,input_steps,prediction_steps)
