import numpy as np
from esnlib import *

np.random.seed(0)
X,y = np.random.rand(10000,24), np.random.rand(10000,12)
X_test = np.random.rand(5,24)
esn = ESN(n_reservoir=5000,random_state=42)
esn.fit(X,y)
print(esn.predict(X_test))
