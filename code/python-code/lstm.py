import argparse

parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('path', help='File path')
parser.add_argument('--inputs', default=1, help="Input Values", type=int)
parser.add_argument('--outputs', default=12, help="Input Values", type=int)
parser.add_argument('--epochs', default=100, help="Input Values", type=int)
parser.add_argument('--preprocess', default="minmax", help="minmax or standarization")
args = parser.parse_args()

#Imports
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as metrics

import sys
sys.path.append('../libs')
from helpers import *
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



def create_lstm(input_dim,output_dim,nodes,loss='mean_squared_error',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(None,input_dim)))
    model.add(Dense(output_dim))
    model.compile(loss=loss, optimizer=optimizer)
    return model

if __name__ == "__main__":
    train_perc = 0.8
    print("Reading Data")
    #Data split parameters
    filename = os.path.basename(args.path).split("_")[1].replace(".csv","")
    X = pd.read_csv(args.path,index_col=0).values[:,:args.inputs]
    if args.outputs == 1:
        y = pd.read_csv(args.path.replace("X", "y"), index_col=0).values[:,0].reshape((-1,1))
    else:
        y = pd.read_csv(args.path.replace("X", "y"), index_col=0).values#[:,args.outputs-1]


    trainlen = int(train_perc*len(X))
    X_train,X_test = X[:trainlen], X[trainlen:]
    y_train,y_test = y[:trainlen], y[trainlen:]
    y_train_orig  = y_train
    print("Preprocessing Data")

    if args.preprocess == "minmax":
    	minmax_in = preprocessing.MinMaxScaler(feature_range=(0,1))
    	minmax_out = preprocessing.MinMaxScaler(feature_range=(0,1))

    	minmax_in.fit(X_train)
    	minmax_out.fit(y_train)

    	preproc_in = minmax_in
    	preproc_out = minmax_out

    else:
    	standarization_in = preprocessing.StandardScaler()
    	standarization_out = preprocessing.StandardScaler()

    	standarization_in.fit(X_train)
    	standarization_out.fit(y_train)

    	preproc_in = standarization_in
    	preproc_out = standarization_out

    X_train = preproc_in.transform(X_train) if preproc_in else X_train
    X_test = preproc_in.transform(X_test) if preproc_in else X_test
    y_train = preproc_out.transform(y_train) if preproc_out else y_train

    X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))

    print("Creating Param List")
    lsm_nodes = [5,10,15,20]
    loss = ["mean_squared_error"]
    param_grid = {"nodes":lsm_nodes,"loss":loss, "input_dim":[args.inputs], "output_dim": [args.outputs]}
    params = ms.ParameterGrid(param_grid)

    print("Evaluating Models")

    scores = []
    for param in params:
        np.random.seed(42)
        model = create_lstm(**param)

        model.fit(X_train, y_train, epochs=args.epochs, batch_size=1)

        y_approx = model.predict(X_train)
        score = metrics.mean_squared_error(y_train,y_approx)

        param["score"] = score
        scores.append(param)
    scores = pd.DataFrame(scores)
    filename = os.path.basename(args.path).split("_")[2]
    scores.to_csv("{}_scores_lstm_{}lags_{}outs.csv".format(filename,args.inputs, args.outputs))
