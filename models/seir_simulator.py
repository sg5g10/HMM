import numpy as np    
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
from CPP import seirssa
#from numba import njit

class SEIRSim(object):
    def __init__(self, simulate_summary=True):
        self._simulate_summary = simulate_summary
        self._simulate_full = False
    
    def simulate(self, params):
        dt = 1  # Time step.
        success = 0
        times = np.arange(0,20,dt)
        z = np.zeros((times.shape[0],4))
        c = np.array(params[1:])
      
        success = 0        
        while (success < 1):
            x_prev = np.array([params[0],0,1,0])
            ss = list()
            for i in range(1,len(times)):
                x_next = np.array(seirssa.SEIR(c, x_prev, times[i-1], times[i]))
                ss.append(x_next)      
                x_prev = x_next
            ss = np.array(ss)
            x0 = np.array([params[0],0,1,0]).reshape((1,4))
            x = np.concatenate((x0, ss),axis=0) 
            z = np.abs(x)
            success += 1

        if self._simulate_summary:
            return z[::5,1].reshape((14))*763
        elif self._simulate_full:
            return z[:,-2:].reshape((20,2)), z
        else:
            return z
    
    def simulate_for_abc(self, params):
        self._simulate_full = True
        self._simulate_summary = False
        summary = []
        full = []
        for i in range(10):
            s, f = self.simulate(params)
            summary.append(np.random.poisson(s)), full.append(f)
        return summary, full

    def __call__(self, parameter_set):
        if self._simulate_summary:
            ar = np.zeros((len(parameter_set),14))
        else:
            ar = np.zeros((len(parameter_set),20,4))
        for i in range(len(parameter_set)):
            ar[i,:] = self.simulate(parameter_set[i])
        return ar