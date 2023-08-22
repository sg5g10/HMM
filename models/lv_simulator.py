import numpy as np    
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
from CPP import lvssa

class LVSim(object):
    def __init__(self, simulate_summary=True):
        self._simulate_summary = simulate_summary
        self._simulate_full = False

    def simulate(self, params):
        sigma = 10
        times = np.arange(0,50,1)
        x = np.zeros((times.shape[0],2))
        c = np.array(params)
        c[1] = c[1]/10000
        
        success = 0
        while (success < 1):
            x_prev = np.array([100,100])
            ss = list()
            for i in range(1,len(times)):
                x_next = np.array(lvssa.LV(c, x_prev, times[i-1], times[i]))
                ss.append(x_next)      
                x_prev = x_next
            ss = np.array(ss)
            x0 = np.array([100,100]).reshape((1,2))
            x = np.concatenate((x0, ss),axis=0) #+ np.random.randn(len(times),2)*sigma
            if np.any(ss<0):
                success += 0
            else:
                success += 1

        if self._simulate_summary:
            return x[::5,:].flatten(order='C')
        elif self._simulate_full:
            return x[::5,:].flatten(order='C'), x
        else:
            return x    

    def simulate_for_abc(self, params):
        self._simulate_full = True
        self._simulate_summary = False
        summary = []
        full = []
        for i in range(10):
            s, f = self.simulate(params)
            summary.append(s + np.random.randn(len(s))*10), full.append(f)
        return summary, full
            
    def __call__(self, parameter_set):    
        if self._simulate_summary:
            ar = np.zeros((len(parameter_set),20))
        else:
            ar = np.zeros((len(parameter_set),50,2))
        for i in range(len(parameter_set)):
            ar[i,:] = self.simulate(parameter_set[i])
        return ar

