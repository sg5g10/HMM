import numpy as np    
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
from CPP import pkyssa

class PKYSim(object):
    def __init__(self, simulate_summary=True):
        self._simulate_summary = simulate_summary
        self._simulate_full = False
    def simulate(self, params):
        sigma = 2
        times = np.arange(0,50,.5)
        x = np.zeros((times.shape[0],4))
        c = np.array(params)
      
        success = 0
        while (success < 1):
            x_prev = np.array([8,8,8,5])
            ss = list()
            for i in range(1,len(times)):
                x_next = np.array(pkyssa.PKY(c, x_prev, times[i-1], times[i]))
                ss.append(x_next)      
                x_prev = x_next
            ss = np.array(ss)
            x0 = np.array([8,8,8,5]).reshape((1,4))
            x = np.concatenate((x0, ss),axis=0) 
            y = (x[:,1]) + (2*x[:,2])
            if np.any(ss<0):
                success += 0
            else:
                success += 1

        if self._simulate_summary:
            return y[::5]
        elif self._simulate_full:
            return y, x
        else:
            return x

    def simulate_for_abc(self, params):
        self._simulate_full = True
        self._simulate_summary = False
        summary = []
        full = []
        for i in range(10):
            s, f = self.simulate(params)
            summary.append(s + np.random.randn(len(s))*2), full.append(f)
        return summary, full
    
    
    def __call__(self, parameter_set):
        if self._simulate_summary:
            ar = np.zeros((len(parameter_set),20))
        else:
            ar = np.zeros((len(parameter_set),100,4))        
        for i in range(len(parameter_set)):
            ar[i,:] = self.simulate(parameter_set[i])
        return ar

