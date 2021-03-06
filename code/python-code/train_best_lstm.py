import argparse

parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('path', help='File path')
parser.add_argument('result_path', help='Results file path')
parser.add_argument('--epochs', default=1, help="Input Values", type=int)
parser.add_argument('--batch', default=1, help="Batch Size", type=int)
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
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


"""
Function to create lstm model
"""
def create_lstm(input_dim,output_dim,nodes,loss='mean_squared_error',optimizer='adam'):
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(None,input_dim)))
    model.add(Dense(output_dim))
    model.compile(loss=loss, optimizer=optimizer)
    return model

#Reading results file
results_df = pd.read_csv(args.result_path,index_col=0)
#Reading data files
data =  pd.read_csv(args.path,index_col=0).values
data_y = pd.read_csv(args.path.replace("X", "y"), index_col=0).values

best_scores = []
for i,result_args in results_df.iterrows():
    input_dim = result_args.input_dim
    output_dim = result_args.output_dim
    #Fixed output dim and changing lags
    X =data[:,:input_dim]
    y = data_y[:,:output_dim]

    arg = result_args.drop(["score","lags","source"])
    train_perc = 0.8
    trainlen = int(train_perc*len(X))
    #Dividing Train and Test
    X_train,X_test = X[:trainlen], X[trainlen:]
    y_train,y_test = y[:trainlen], y[trainlen:]
    y_train_orig  = y_train

    #Using 0,1
    minmax_in = preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax_out = preprocessing.MinMaxScaler(feature_range=(0,1))

    minmax_in.fit(X_train)
    minmax_out.fit(y_train)

    preproc_in = minmax_in
    preproc_out = minmax_out

    #Transforming data
    X_train = preproc_in.transform(X_train) if preproc_in else X_train
    X_test = preproc_in.transform(X_test) if preproc_in else X_test
    y_train = preproc_out.transform(y_train) if preproc_out else y_train

    X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))

    #Creating model
    model = create_lstm(**arg)
    t0 = time.time()
    model.fit(X_train,y_train,epochs=args.epochs, batch_size=args.batch)
    t1 = time.time() - t0
    #Predicting Train
    y_approx = model.predict(X_train)
    y_approx = preproc_out.inverse_transform(y_approx)
    score_train = metrics.mean_squared_error(y_train_orig,y_approx)

    #Predicting Test
    y_test_approx = model.predict(X_test)
    y_test_approx = preproc_out.inverse_transform(y_test_approx)
    score_test = metrics.mean_squared_error(y_test,y_test_approx)

    best_scores.append({"score_test":score_test, "score_train": score_train, "lags": result_args.lags, "time": t1})
    model.save(result_args.source + "_{}lags_model.h5".format(result_args.lags))
    np.savetxt("lstm_y_train_{}.csv".format(result_args.source),y_train_orig)
    np.savetxt("lstm_y_test_{}.csv".format(result_args.source),y_test)

    np.savetxt("lstm_y_approx_train_{}lags_{}.csv".format(result_args.lags, result_args.source),y_approx)
    np.savetxt("lstm_y_approx_test_{}lags_{}.csv".format(result_args.lags, result_args.source),y_test_approx)

best_scores = pd.DataFrame(best_scores)
best_scores.to_csv("lstm_{}_best_scores.csv".format(result_args.source))
