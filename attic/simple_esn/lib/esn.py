import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

class ESN(BaseEstimator, RegressorMixin):

    def __init__(self, n_hidden=100, sparsity = 0.95, leaking_rate=0.3, random_state = None, regularization =1e-8, activation = None, spectral_radius = None, verbose = False):
        self.n_hidden = n_hidden
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.random_state = random_state
        self.verbose = verbose
        self.regularization = regularization
        self.activation = activation if activation else np.tanh
        self.spectral_radius = spectral_radius

    def fit(self,X,y):

        X = X.T 
        y = y.T if len(y.shape) > 1 else y.reshape((-1,y.shape))

        self.inSize = X.shape[0]

        self.outSize = y.shape[0]

        trainLen = X.shape[1]

        resSize = self.n_hidden
        a = self.leaking_rate # leaking rate


        if self.random_state:
            np.random.seed(self.random_state)
        

        self.Win = (np.random.rand(resSize,1+self.inSize)-0.5) * 1
        self.Win[np.random.rand(*self.Win.shape) < self.sparsity] = 0


        self.W = np.random.rand(resSize,resSize)-0.5 
        #self.W[np.random.rand(*self.W.shape) < self.sparsity] = 0



        if not(self.spectral_radius):
            if self.verbose:
                print('Computing spectral radius...')
        
            rhoW = np.max(np.abs(np.linalg.eig(self.W)[0]))
        
            if self.verbose:
                print('done.')

            self.W *= 1.25 / rhoW
        else:
            self.W *= self.spectral_radius

        
        CS = np.zeros((1+self.inSize+resSize,trainLen))
        Yt = y

        self.x = np.zeros((resSize,1))
 
 
        u = X[:,0].reshape((self.inSize,1))


        self.x = (1-a)*self.x + a*self.activation( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ))

        CS[:,0] = np.vstack((1,u,self.x))[:,0]
        for t in range(1,trainLen):
            u = X[:,t].reshape((self.inSize,1))
            self.x = (1-a)*self.x + a*self.activation( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ))
            CS[:,t] = np.vstack((1,u,self.x))[:,0]

        # train the output
        reg = self.regularization  # regularization coefficient
        CS_T = CS.T

        self.Wout = np.dot( np.dot(Yt,CS_T), np.linalg.inv( np.dot(CS,CS_T) + \
            reg*np.eye(1+self.inSize+resSize) ) )

        #self.Wout = np.dot( Yt, np.linalg.pinv(CS) )

        return self
            

    def predict(self,X):
        # Check is fit had been called
        check_is_fitted(self, ['W', 'Win','Wout'])

        X = X.T 
        
        X = check_array(X)
        testLen = X.shape[1]

        a = self.leaking_rate

        Y = np.zeros((self.outSize,testLen))
        u = X[:,0].reshape((self.inSize,1))
        for t in range(testLen):
            self.x = (1-a)*self.x + a*self.activation( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            Y[:,t] = y[:,0]
            # generative mode:
            
            if np.isnan(y).any():
                raise Exception("u:{}".format(u))

            u = X[:,t].reshape((self.inSize,1))

         
        return Y.T



if __name__ == "__main__":
    from utils import getDataWindowed

    import matplotlib.pyplot as plt
    data = np.loadtxt('../MackeyGlass_t17.txt')

    input_steps = 24
    prediction_steps =1

    X,y = getDataWindowed(data,input_steps,prediction_steps)
    esn = ESN(verbose=True, n_hidden=100, spectral_radius=0.1, leaking_rate=0.1, regularization=1e-5)
    esn.fit(X,y)
    y_pred = esn.predict(X)

    plt.plot(y[1:200,0],'r')
    plt.plot(y_pred[1:200,0],'g')
    plt.show()