import numpy as np
import pandas as pd
from numba import jit
import sklearn.metrics as sm
#
# @jit
# def getDataWindowed(data,inSize,outSize):
#     biggest = np.max([inSize,outSize])
#
#     matrixIn = np.zeros((len(data)-2*biggest, inSize))
#     matrixOut = np.zeros((len(data)-2*biggest, outSize))
#     for i in range(len(data)-2*biggest):
#         matrixIn[i,:] = data[i:i+inSize]
#         matrixOut[i,:] = data[i+inSize+1:i+inSize+outSize+1]
#     return matrixIn,matrixOut
@jit
def getDataWindowed(data,inSize,outSize):
    biggest = np.max([inSize,outSize])

    if len(data.shape) > 1:
        N, dims = data.shape
    else:
        N = len(data)
        dims = 1

    matrixIn = np.zeros((N-2*biggest, inSize*dims))
    matrixOut = np.zeros((N-2*biggest, outSize))
    for i in range(N-2*biggest):
        if len(data.shape) > 1:
            matrixIn[i,:] = data[i:i+inSize,:].flatten()
            matrixOut[i,:] = data[i+inSize+1:i+inSize+outSize+1,0]
        else:
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


def show_errors(y_test,y_pred):
    if type(y_test) is pd.DataFrame:
        y_test = y_test.values
    if type(y_pred) is pd.DataFrame:
        y_pred = y_pred.values

    if len(y_test.shape) > 1:
        _, n_steps = y_test.shape
    else:
        n_examples = len(y_test)
        n_steps = 1
        y_test = y_test.reshape((n_examples,1))
        y_pred = y_pred.reshape((n_examples,1))

    n_metrics = 5
    metrics = np.zeros((n_metrics,n_steps+1))
    for i in range(n_steps):
        mae = sm.mean_absolute_error(y_test[:,i],y_pred[:,i])
        mse = sm.mean_squared_error(y_test[:,i],y_pred[:,i])
        medae = sm.median_absolute_error(y_test[:,i],y_pred[:,i])
        r2 = sm.r2_score(y_test[:,i],y_pred[:,i])
        mape = mape_score(y_test[:,i],y_pred[:,i])
        metrics[0,i] = mae
        metrics[1,i] = mse
        metrics[2,i] = medae
        metrics[3,i] = mape
        metrics[4,i] = r2
    for i in range(n_metrics):
        metrics[i,n_steps] = np.mean(metrics[i,:n_steps])

    metrics = pd.DataFrame(metrics)
    column_names = []
    for i in range(n_steps):
        column_names.append("$t+{}$".format(i))
    column_names.append("$\overline{t+i}$")
    metrics.index = ["MAE","MSE", "MeAE", "MAPE" ,"$r^2$"]
    metrics.columns = column_names
    return metrics

#activation
def relu(x):
    return x * (x > 0)
