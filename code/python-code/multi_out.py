import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('path', help='File path')
parser.add_argument('--inputs', default=1, help="Input Values", type=int)
parser.add_argument('--preprocess', default="minmax", help="minmax or standarization")
args = parser.parse_args()

#IMPORTS
import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import multiprocessing as mp

import pickle

import sys
sys.path.append('../libs')
from esnlib import *
from helpers import *
import os

def worker(args):
	i,param, X_train,y_train,y_train_orig,preproc_out= args
	print("training {}".format(i))
	clf = ESN(random_state=42, **param)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_train)
	y_pred = preproc_out.inverse_transform(y_pred)
	score = metrics.mean_squared_error(y_train_orig,y_pred)
	return score

if __name__ == "__main__":
	train_perc = 0.8


	print("Reading Data")
	#Data split parameters
	filename = os.path.basename(args.path).split("_")[2]
	X = pd.read_csv(args.path,index_col=0).values[:,:args.inputs]
	y = pd.read_csv(args.path.replace("x_potency", "y"), index_col=0).values
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

	print("Creating Param List")
	n_reservoir = 100
	n_params= 6
	sparsity = [0.2,0.5,0.9]
	leaking_rate = np.linspace(0.3,0.9,5)
	regularization= np.logspace(4,-4,base=10,num=n_params)
	spectral_radius= np.logspace(-5,0,base=10,num=n_params)
	param_grid = {"n_reservoir":[n_reservoir], "sparsity":sparsity, "leaking_rate":leaking_rate, "regularization":regularization, "spectral_radius":spectral_radius}
	params = ms.ParameterGrid(param_grid)

	print("Evaluating Models")
	param_list = []
	for i,param in enumerate(params):
		param_list.append((i,param, X_train,y_train,y_train_orig,preproc_out))
	pool = mp.Pool(mp.cpu_count())
	scores = pool.map(worker,param_list)


	param_df = pd.DataFrame(list(params))
	param_df["scores"] = scores
	param_df.to_csv("{}_scores_multi_{}lags.csv".format(filename,args.inputs))
