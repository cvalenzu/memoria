import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import sys
sys.path.append('../libs')

from esnlib import *

# fix random seed for reproducibility
numpy.random.seed(42)
data = pd.read_csv("../../data/canela1_merged.csv")

X,y=getDataWindowed(data.ws.as_matrix(),24,12)

data_len = len(X)
train_perc = 0.8
train_len = int(data_len * train_perc)

X_train, y_train = X[:train_len], y[:train_len]
X_test, y_test = X[train_len:], y[train_len:]
X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))

model = Sequential()
model.add(LSTM(32, input_dim=24))
model.add(Dense(12))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1)

y_approx = model.predict(X_test)

np.savetxt("y_approx.csv",y_approx,delimiter=",")
np.savetxt("y_test.csv",y_test,delimiter=",")
