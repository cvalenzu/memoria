import sys
sys.path.append('../libs')
from esnlib import *
from helpers import *
import pandas as pd
import sklearn.metrics as metrics
from scipy.special import expit
import pickle
from sklearn import preprocessing
import sklearn.model_selection as ms
import os
os.makedirs("results_one",exist_ok=True)


#Loading data
print("Reading Data")
data = pd.read_csv('../../data/canela1_merged.csv',index_col=0).values

#Data split parameters
input_steps = 1
prediction_steps = 1
train_perc = 0.8

X,y = getDataWindowed(data,input_steps,prediction_steps)

trainlen = int(train_perc*len(X))
X_train,X_test = X[:trainlen], X[trainlen:]
y_train,y_test = y[:trainlen], y[trainlen:]

#Creating Time Series Validation Folds
print("Creating Folds")
n_splits = 5
tscv = ms.TimeSeriesSplit(n_splits=n_splits)

#Reading processed ones
if os.path.exists("results_one/process_index.csv"):
    processed = np.loadtxt("results_one/process_index.csv")
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


#Metrics
def scorer(estimator, X,y):
    y_pred = estimator.predict(X,cont=True)
    n_steps = y.shape[1] if len(y.shape) > 1 else 1
    if len(y.shape) <= 1:
        N = len(y)
        y = y.reshape((N,1))
        y_pred = y_pred.reshape((N,1))

    scores = []
    for i in range(n_steps):
        scores.append(metrics.mean_squared_error(y[:,i],y_pred[:,i]))
    scores = np.array(scores)
    return scores.mean()

print("Creating Param List")
#PARAMS
n_reservoir = 1000
sparsity = [0,0.5,0.9]#np.linspace(0.5,0.9,3)
leaking_rate = [1,0.5,0.1]#np.linspace(0.3,0.9,3)
regularization= [0,1e-5,1,2]#[1e-8,1e-5,1e-2]
spectral_radius= [1e-8,0.1,1,2]#np.linspace(1e-5,1,3)
activation = [np.tanh,expit]
param_grid = {"n_reservoir":[n_reservoir], "sparsity":sparsity, "leaking_rate":leaking_rate, "regularization":regularization, "activation": activation, "spectral_radius":spectral_radius}
params = ms.ParameterGrid(param_grid)

with open('results_one/params.pkl', 'wb') as fout:
    pickle.dump(list(params), fout)

print("Evaluating Models")
for i,param in enumerate(params):
    if i not in processed:
        clf = ESN(random_state=42, **param)
        print(clf.get_params())
        score = ms.cross_val_score(clf,X_train,y_train, cv = tscv, n_jobs=-1,scoring=scorer)
        with open("results_one/process_index.csv","a+") as process_index:
            process_index.write("{}\n".format(i))
            with open("results_one/scores.csv", "a+") as scores_file:
                scores_file.write('{},'.format(i)+','.join([str(num) for num in score]) + "\n")
                scores_file.close()
            process_index.close()

        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_pred = preproc_out.inverse_transform(y_pred)
