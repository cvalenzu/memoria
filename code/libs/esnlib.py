import os
import pandas as pd
import numpy as np
from numba import jit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


@jit
def getDataWindowed(data,inSize,outSize):
    biggest = np.max([inSize,outSize])

    matrixIn = np.zeros((len(data)-2*biggest, inSize))
    matrixOut = np.zeros((len(data)-2*biggest, outSize))
    for i in range(len(data)-2*biggest):
        matrixIn[i,:] = data[i:i+inSize]
        matrixOut[i,:] = data[i+inSize+1:i+inSize+outSize+1]
    return matrixIn,matrixOut

class ESN(BaseEstimator,RegressorMixin):
    def __init__(self, n_reservoir = 1000, spectral_radius = 0.135, sparsity=0,
                 leaking_rate=0.3, regularization=1, random_state = None, activation = np.tanh):
        self.n_inputs = None
        self.n_outputs = None
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.regularization = regularization
        self.last_state = None
        self.activation = activation
        if random_state:
            if type(random_state) is int:
                self.random_state=np.random.RandomState(random_state)
            elif type(random_state) is np.random.RandomState:
                self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

    def get_params(self,deep=True):
        params =  {'n_reservoir':self.n_reservoir,'spectral_radius':self.spectral_radius,  'sparsity':self.sparsity,
                  'leaking_rate': self.leaking_rate, "regularization":self.regularization, "activation": self.activation}
        if self.n_inputs and self.n_outputs:
            params["n_inputs"] = self.n_inputs
            params["n_outputs"] = self.n_outputs
        if self.random_state:
            params["random_state"] = self.random_state
        return params

    @jit
    def fit(self,X,y):
        in_rows,self.n_inputs = X.shape
        if len(y.shape) > 1:
            out_rows, self.n_outputs = y.shape
        else:
            out_rows = len(y)
            self.n_outputs = 1
        initLen = int(0.01*in_rows)

        #Raise exception
        assert(in_rows == out_rows)

        #Input length
        N = in_rows

        #Creating input weights
        self.Win = (self.random_state.rand(self.n_reservoir,1+self.n_inputs)-0.5) * 1

        #Creating Reservoir weights
        self.W = self.random_state.rand(self.n_reservoir,self.n_reservoir)-0.5
        #Sparsity
        self.W[self.random_state.rand(*self.W.shape) < self.sparsity] = 0
        #Spectral radius
        self.W *= self.spectral_radius

        #Creating state matrix
        X_states = np.zeros((1+self.n_inputs+self.n_reservoir,N-initLen))

        #Last state
        self.last_state  = np.zeros(self.n_reservoir)

        #Collecting states
        for t in range(N):
            u = X[t]
            #Calculating new state
            self.last_state = (1-self.leaking_rate)*self.last_state  + self.leaking_rate*self.activation( np.dot( self.Win, np.hstack((1,u)) ) \
                                                                    + np.dot( self.W, self.last_state  ) )
            if t >= initLen:
                X_states[:,t-initLen] = np.hstack((1,u,self.last_state ))


        Y_T = y[initLen:].T

        X_sqrd = np.dot(X_states,X_states.T)+  self.regularization*np.eye(1+self.n_inputs+self.n_reservoir)
        Y_sqrd = np.dot(Y_T,X_states.T)

        #Getting the output weights using least squares
        self.Wout=np.dot(Y_sqrd, np.linalg.inv(X_sqrd))

        return self

    @jit
    def predict(self,X, cont=False):
        Y = np.empty((len(X),self.n_outputs))
        if not cont:
            last_state = np.zeros_like(self.last_state)
        else:
            last_state = self.last_state
        for t,u in enumerate(X):
                last_state = (1 - self.leaking_rate) * last_state + self.leaking_rate*self.activation( np.dot( self.Win, np.hstack((1,u)))+ \
                                                                     + np.dot( self.W, last_state  ) )
                y = np.dot(self.Wout, np.hstack((1,u,last_state)))
                Y[t,:] = y

        if cont:
            self.last_state = last_state
        return Y
