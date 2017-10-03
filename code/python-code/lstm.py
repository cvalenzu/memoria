import sys
sys.path.append('../libs')
from helpers import *
import pandas as pd
import sklearn.metrics as metrics
from scipy.special import expit
import pickle
from sklearn import preprocessing
import sklearn.model_selection as ms
import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

os.makedirs("results_lstm",exist_ok=True)

from esnlib import *

n_epochs = 1
n_splits = 2

def create_lstm(input_dim,nodes,loss='mean_squared_error',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(nodes, input_dim=input_dim))
    model.add(Dense(12))
    model.compile(loss=loss, optimizer=optimizer)
    return model

#Loading data
print("Reading Data")
data = pd.read_csv('../../data/canela1_merged.csv',index_col=0).values

#Data split parameters
input_steps_list = [6,12,24,48,72]
prediction_steps = 12
train_perc = 0.8

for input_steps in input_steps_list:
    print("Processing data with input_steps= {}".format(input_steps))
    os.makedirs("results_lstm/{}".format(input_steps),exist_ok=True)
    X,y=getDataWindowed(data,input_steps,prediction_steps)
    trainlen = int(train_perc*len(X))
    X_train,X_test = X[:trainlen], X[trainlen:]
    y_train,y_test = y[:trainlen], y[trainlen:]

    #Creating Time Series Validation Folds
    print("Creating Folds")

    tscv = ms.TimeSeriesSplit(n_splits=n_splits)

    #Reading processed ones
    if os.path.exists("results_lstm/{}/process_index.csv".format(input_steps)):
        processed = np.loadtxt("results_lstm/{}/process_index.csv".format(input_steps))
    else:
        processed = []

    #Preprocess data
    print("Preprocessing Data")
    minmax_in = preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax_out = preprocessing.MinMaxScaler(feature_range=(0,1))
    standarization_in = preprocessing.StandardScaler()
    standarization_out = preprocessing.StandardScaler()

    minmax_in.fit(X_train)
    minmax_out.fit(y_train)
    standarization_in.fit(X_train)
    standarization_out.fit(X_train)
    preproc_in = minmax_in
    preproc_out = minmax_out

    X_train = preproc_in.transform(X_train) if preproc_in else X_train
    X_test = preproc_in.transform(X_test) if preproc_in else X_test

    y_train = preproc_out.transform(y_train) if preproc_out else y_train

    print("Creating Param List")
    lsm_nodes = [5,10,15,20]
    optimizer = ["sgd", "adam","adadelta"]
    loss = ["mean_squared_error"]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    input_dim = [input_steps*dims]
    param_grid = {"nodes":lsm_nodes, "optimizer":optimizer, "loss":loss, "input_dim":input_dim}
    params = ms.ParameterGrid(param_grid)

    with open('results_lstm/{}/params.pkl'.format(input_steps), 'wb') as fout:
    	pickle.dump(list(params), fout)

    print("Evaluating Models")
    for i,param in enumerate(params):
    	if i not in processed:
            np.random.seed(42)
            model = create_lstm(**param)
            scores = []
            for split_train,split_val in tscv.split(X_train):
                X_ttrain, X_val = X_train[split_train,:],  X_train[split_val,:]
                y_ttrain, y_val = y_train[split_train,:],  y_train[split_val,:]
                X_ttrain = X_ttrain.reshape((X_ttrain.shape[0],1,X_ttrain.shape[1]))
                X_val = X_val.reshape((X_val.shape[0],1,X_val.shape[1]))

                model.fit(X_ttrain, y_ttrain, epochs=n_epochs, batch_size=1)

                y_approx = model.predict(X_val)
                score = metrics.mean_squared_error(y_val,y_approx)
                scores.append(score)
            with open("results_lstm/{}/process_index.csv".format(input_steps),"a+") as process_index:
                process_index.write("{}\n".format(i))
                with open("results_lstm/{}/scores.csv".format(input_steps), "a+") as scores_file:
                    scores_file.write('{},'.format(i)+','.join([str(num) for num in scores]) + "\n")
                    scores_file.close()
                process_index.close()
