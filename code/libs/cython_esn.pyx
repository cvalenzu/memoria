import numpy as np
cimport numpy as np


cdef _collect_states(X,activation,Win, W,leaking_rate, initLen, n_reservoir,n_inputs):
  N = len(X)
  last_state  = np.zeros(n_reservoir)
  X_states = np.zeros((N-initLen,1+n_inputs+n_reservoir))

  #Collecting states
  for t in range(N):
      u = X[t]
      #Calculating new state
      last_state = (1-leaking_rate)*last_state  + leaking_rate*activation( np.dot( Win, np.hstack((1,u)) ) \
                                                            + np.dot( W, last_state  ) )
      if t >= initLen:
        X_states[t-initLen,:] = np.hstack((1,u,self.last_state ))
