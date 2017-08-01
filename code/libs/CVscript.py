from esnlib import *
import pandas as pd

from sklearn import preprocessing
import sklearn.metrics as metrics

import sklearn.model_selection as ms
import matplotlib.pyplot as plt

import pickle

import argparse

n_reservoir = 10000
n_linspace = 5
n_splits = 3

parser = argparse.ArgumentParser(description='Cross Validation ESN Meteo')
parser.add_argument('output',help=["Output File Name"])
parser.add_argument('-p','--preproc',help=["Preproc used"])

args = parser.parse_args()

print("Loading data")
data = pd.Series.from_csv('../../data/meteo.csv')

prediction_steps = 12
input_steps = 24
train_perc = 0.6

print("Generating X,y")
X,y = getDataWindowed(data,input_steps,prediction_steps)

trainlen = int(train_perc*len(X))
X_train,X_test = X[:trainlen], X[trainlen:]
y_train,y_test = y[:trainlen], y[trainlen:]


minmax_in = preprocessing.MinMaxScaler(feature_range=(-1,1))
standarization_in = preprocessing.StandardScaler()
minmax_out = preprocessing.MinMaxScaler(feature_range=(-1,1))
standarization_out = preprocessing.StandardScaler()

minmax_in.fit(X_train)
minmax_out.fit(y_train)
standarization_in.fit(X_train)
standarization_out.fit(X_train)

print("Preprocessing")
if args.preproc == "std":
    preproc_in = standarization_in
    preproc_out = standarization_out
elif args.preproc == "minmax":
    preproc_in = minmax_in
    preproc_out = minmax_out
else:
    preproc_in = None
    preproc_out = None

X_train = preproc_in.transform(X_train) if preproc_in else X_train
X_test = preproc_in.transform(X_test) if preproc_in else X_test

y_train = preproc_out.transform(y_train) if preproc_out else y_train
y_test = preproc_out.transform(y_test) if preproc_out else y_test

def scorer(estimator, X,y):
    y_pred = estimator.predict(X,cont=True)
    #return metrics.r2_score(y,y_pred)
    return -metrics.mean_squared_error(y,y_pred)
print("Creating CV")
tscv = ms.TimeSeriesSplit(n_splits=n_splits)
clf = ESN(random_state=42,n_reservoir=n_reservoir)
params = {'regularization':np.linspace(1e-8,1,n_linspace),'spectral_radius':np.linspace(0,1,n_linspace),
            'sparsity':np.linspace(0.1,0.99,n_linspace), 'leaking_rate':np.linspace(0,1,n_linspace)}
cv = ms.GridSearchCV(clf,params, n_jobs=-1, cv=tscv, verbose=True, scoring=scorer, iid=False)
cv.fit(X_train,y_train)

print("Saving CV results")
pickle.dump(cv, open("cv_{}.p".format(args.output), "wb"))

clf.set_params(**cv.best_params_)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Saving Best Data")
pickle.dump(clf, open("best_esn_{}.p".format(args.output),"wb"))
pickle.dump(y_pred, open("y_pred_{}.p".format(args.output),"wb"))
pickle.dump(y_test,open("y_test_{}.p".format(args.output),"wb"))
