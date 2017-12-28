import argparse

parser = argparse.ArgumentParser(description='Run LSTM.')
parser.add_argument('path', help='File path')
parser.add_argument('result_path', help='Results file path')
parser.add_argument('--epochs', default=1, help="Input Values", type=int)
parser.add_argument('--reservoir', default=100, help="Input Values", type=int)
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
from esnlib import *
import os
import time
import pickle

results_df = pd.read_csv(args.result_path,index_col=0)
data =  pd.read_csv(args.path,index_col=0).values
data_y = pd.read_csv(args.path.replace("X", "y"), index_col=0).values
best_scores = []

for i,result_args in results_df.iterrows():
    input_dim = result_args.lags
    output_dim = 12
    X =data[:,:input_dim]
    y = data_y[:,:output_dim]

    arg = result_args.drop(["scores","lags","source","n_reservoir"])
    train_perc = 0.8
    trainlen = int(train_perc*len(X))
    X_train,X_test = X[:trainlen], X[trainlen:]
    y_train,y_test = y[:trainlen], y[trainlen:]
    y_train_orig  = y_train

    minmax_in = preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax_out = preprocessing.MinMaxScaler(feature_range=(0,1))

    minmax_in.fit(X_train)
    minmax_out.fit(y_train)

    preproc_in = minmax_in
    preproc_out = minmax_out

    X_train = preproc_in.transform(X_train) if preproc_in else X_train
    X_test = preproc_in.transform(X_test) if preproc_in else X_test
    y_train = preproc_out.transform(y_train) if preproc_out else y_train

    n_reservoir = args.reservoir
    clf = ESN(random_state=42,n_reservoir=n_reservoir, **arg)
    t0 = time.time()
    clf.fit(X_train,y_train)
    t1 = time.time()-t0


    y_pred = clf.predict(X_train)
    y_pred = preproc_out.inverse_transform(y_pred)

    score_train = metrics.mean_squared_error(y_train_orig,y_pred)

    y_pred_test = clf.predict(X_test,cont=True)
    y_pred_test = preproc_out.inverse_transform(y_pred_test)
    score_test = metrics.mean_squared_error(y_test,y_pred_test)

    best_scores.append({"score_test":score_test, "score_train": score_train, "lags": result_args.lags, "time": t1})
    with open("{}_{}lags_esn_model.pkl".format(result_args.source,result_args.lags),"wb") as write_file:
        pickle.dump(clf, write_file)
    np.savetxt("esn_y_train_{}.csv".format(result_args.source),y_train_orig)
    np.savetxt("esn_y_test_{}.csv".format(result_args.source),y_test)

    np.savetxt("esn_y_approx_train_{}lags_{}.csv".format(result_args.lags, result_args.source),y_pred)
    np.savetxt("esn_y_approx_test_{}lags_{}.csv".format(result_args.lags, result_args.source),y_pred_test)

best_scores = pd.DataFrame(best_scores)
best_scores.to_csv("esn_{}_best_scores.csv".format(result_args.source))
