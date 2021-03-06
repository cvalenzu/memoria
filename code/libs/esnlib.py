import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
# from cython_esn import _collect_states
from numba import jit
class ESN(BaseEstimator,RegressorMixin):
    def __init__(self, n_reservoir = 1000, spectral_radius = 0.135, sparsity=0,
                 leaking_rate=0.3, regularization=1, random_state = None, activation = "tanh"):
        self.n_inputs = None
        self.n_outputs = None
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.regularization = regularization
        self.last_state = None
        if activation == "tanh":
            self.activation = np.tanh
        elif activation == "sigmoid":
            self.activation = lambda x: 1/(1+np.exp(-1))
        elif activation == "relu":
            self.activation = lambda x: np.maximum(0,x)

        if random_state:
            if type(random_state) is int:
                self.random_state=np.random.RandomState(random_state)
            elif type(random_state) is np.random.RandomState:
                self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()
        self.ridge = Ridge(alpha=regularization, fit_intercept=False, tol=1e-8)

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
        self.Win = self.random_state.uniform(-1,1,(self.n_reservoir,1+self.n_inputs))

        #Creating Reservoir weights
        self.W = self.random_state.uniform(-1,1,(self.n_reservoir,self.n_reservoir))
        #Sparsity
        self.W[self.random_state.rand(*self.W.shape) < self.sparsity] = 0
        #Spectral radius
        spectral_radius = np.max(np.absolute(np.linalg.eigvals(self.W)))
        self.W /= spectral_radius
        self.W *= self.spectral_radius

        # X_states,self.last_state = _collect_states(X,self.activation,self.Win, self.W,self.leaking_rate, initLen, self.n_reservoir,self.n_inputs)
        #Creating state matrix
        X_states = np.zeros((N-initLen,1+self.n_inputs+self.n_reservoir))

        #Last state
        self.last_state  = np.zeros(self.n_reservoir)

        #Collecting states
        for t in range(N):
            u = X[t]
            #Calculating new state
            self.last_state = (1-self.leaking_rate)*self.last_state  + self.leaking_rate*self.activation( np.dot( self.Win, np.hstack((1,u)) ) \
                                                                    + np.dot( self.W, self.last_state  ) )
            if t >= initLen:
                X_states[t-initLen,:] = np.hstack((1,u,self.last_state ))

        Y = y[initLen:]
        #Getting the output weights using least squares
        self.ridge.fit(X_states,Y)
        return self


    def predict(self,X, cont=False):
        Y = np.empty((len(X),self.n_outputs))
        if not cont:
            last_state = np.zeros_like(self.last_state)
        else:
            last_state = self.last_state
        for t,u in enumerate(X):
                last_state = (1 - self.leaking_rate) * last_state + self.leaking_rate*self.activation( np.dot( self.Win, np.hstack((1,u)))+ \
                                                                     + np.dot( self.W, last_state  ) )
                y = self.ridge.predict( np.hstack((1,u,last_state)).reshape(1,-1))
                Y[t,:] = y

        if cont:
            self.last_state = last_state
        return Y

    def n_predict(self,X, n = 12, cont=False):
        Y = np.empty((len(X),n))
        if not cont:
            last_state = np.zeros_like(self.last_state)
        else:
            last_state = self.last_state
        for t,u in enumerate(X):

            for i in range(n):
                last_state = (1 - self.leaking_rate) * last_state + self.leaking_rate*self.activation( np.dot( self.Win, np.hstack((1,u)))+ \
                                                                     + np.dot( self.W, last_state  ) )
                y = self.ridge.predict( np.hstack((1,u,last_state)).reshape(1,-1))
                Y[t,i] = y
                if i == 0:
                    save_last = last_state
            last_state = save_last
        if cont:
            self.last_state = last_state
        return Y
