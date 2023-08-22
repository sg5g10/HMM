import torch
from torch.distributions import Gamma
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
from models.pky_simulator import PKYSim
from CPP import pkyssa
from neural_net.train_nflow import train

def BootStrapp(y, param, P, times):
    D = 4
    c = np.array(param)
    num_steps = len(y) - 1
    x_0 = np.ones((P,D))*[8.,8.,8.,5.]
    z = np.zeros((P,D))
    X = np.zeros((num_steps,P,D))

    w = np.exp(stats.norm(x_0[:,1] + 2*x_0[:,2],2).logpdf(y[0]))
    wt = w/np.sum(w)
    mLik = np.log(w.mean())
    
    ind = np.random.choice(P,size=(P,),p=wt)
    particles = x_0[ind,:]
    t = np.zeros(num_steps)
    
    for i in range(num_steps):      
        z = particles
        for p in range(P):
            z[p] = np.array(pkyssa.PKY(c, z[p], times[i], times[i+1]))   
    
        
        w = np.exp(stats.norm(z[:,1] + 2*z[:,2],2).logpdf(y[i+1]))
        mLik += np.log(w.mean())
        wt = w/np.sum(w)
        particles = z
        ind = np.random.choice(P,size=(P,),p=wt)
        particles = particles[ind,:]
        X[i,:,:] = particles
    x0 = np.array([8.,8.,8.,5.]).reshape((1,4)) 
    return np.concatenate((x0, X[:,0,:]),axis=0)

time = 100
num_sim = 5000
d = 4

_simulator = PKYSim(simulate_summary=False)
gen_par = np.array([[0.1,0.7,0.35,0.2,0.1,0.9,0.3,0.1]])
x_gen = _simulator(gen_par).squeeze()
param_filename = './results/pky_paths.p'
pickle.dump(x_gen, open(param_filename, 'wb'))
y = (x_gen[:,1] + (2*x_gen[:,2])) + (np.random.randn(time).astype(np.float32)*2)
y = y.reshape((-1,1))
param_filename = './results/pky_data.p'
pickle.dump(y, open(param_filename, 'wb'))

prior = Gamma(torch.tensor([2.]), torch.tensor([3.])).expand([8])
theta = prior.sample_n(num_sim)


x = torch.from_numpy(_simulator(theta))
 
noisy_x = (x[:,:,1] + (2*x[:,:,2])) + (np.random.randn(num_sim,time).astype(np.float32)*2)

species, ys, constants = x.detach().numpy(), noisy_x.detach().numpy().reshape((-1,time,1)), theta.detach().numpy()
species = species + np.random.rand(*species.shape)

Path=np.concatenate((species,np.roll(ys,-1,axis=1),constants.reshape((num_sim,1,8)).repeat(time,axis=1)),axis=2)
path = np.array(Path, dtype=np.float32)
x_next = np.concatenate(np.split(path[:num_sim,1::2,:4],int(time/2),axis=1),axis=0).squeeze()
x_prev = np.concatenate(np.split(path[:num_sim,0::2,:],int(time/2),axis=1),axis=0).squeeze() 
 

x_next_ = torch.from_numpy(x_next)
x_prev_ = torch.from_numpy(x_prev)
density_estimator = train(x_next_, x_prev_)

Path_gibbs=np.concatenate((species,np.roll(species,-2,axis=1),np.roll(ys,-1,axis=1),\
    constants.reshape((num_sim,1,8)).repeat(time,axis=1)),axis=2)
path_gibbs = np.array(Path_gibbs[:,:-2,:], dtype=np.float32)
x_next = np.concatenate(np.split(path_gibbs[:,1::4,:4],int(time/4),axis=1),axis=0).squeeze()
x_prev = np.concatenate(np.split(path_gibbs[:,0::4,:],int(time/4),axis=1),axis=0).squeeze()  

x_next_gibbs = torch.from_numpy(x_next)
x_prev_gibbs = torch.from_numpy(x_prev)
density_estimator_gibbs = train(x_next_gibbs, x_prev_gibbs)
      

T=y.shape[0]
P=5000
y_new = y
y_rep=np.array([y_new]*P)




theta_rep = np.array([gen_par[0,:]]*P).reshape((P,8))
x_ = np.zeros((P, T,d))
x_[:,0,:] = [8.,8.,8.,5.]
for n in range(1,T):
    cond = torch.from_numpy(np.hstack((x_[:,n-1,:],y_rep[:,n,:],theta_rep)).astype(np.float32)).view((-1,13))
    x_[:,n,:] = density_estimator.sample(1,cond).detach().squeeze()

x_tch = torch.from_numpy(x_.astype(np.float32))
w = np.zeros((T,P))
x__= x_.copy()
x_p = np.zeros((P, T,d))
x_p[:,0,:] = x__[:,0,:]
x_p[:,T-1,:] = x__[:,T-1,:]
for n in range(T-2,0,-1):
    cond_gibbs = torch.from_numpy(np.hstack((x_[:,n-1,:],x_[:,n+1,:],y_rep[:,n,:],theta_rep)).astype(np.float32)).view((-1,17))
    cond = torch.from_numpy(np.hstack((x_[:,n-1,:],y_rep[:,n,:],theta_rep)).astype(np.float32)).view((-1,13))
    w[n,:] = np.exp(density_estimator_gibbs.log_prob(x_tch[:,n,:].view((-1,d)),cond_gibbs).detach().numpy()
    - density_estimator.log_prob(x_tch[:,n,:].view((-1,d)),cond).detach().numpy())
    w[n,:] = w[n,:]/np.sum(w[n,:])
    ind = np.random.choice(P,size=(P,),p=w[n,:])
    x_p[:,n,:] = x__[ind,n,:]
x_gibbs = np.floor(x_p[0])
param_filename = './results/pky_gibbs.p'
pickle.dump(x_gibbs, open(param_filename, 'wb'))

times = np.arange(0,50,.5)
x_smc = BootStrapp(y, np.array(gen_par[0,:]), 5000, times)
param_filename = './results/pky_smc.p'
pickle.dump(x_smc, open(param_filename, 'wb'))

plt.plot(x_gibbs[:,1] + (2*x_gibbs[:,2]))
plt.plot(x_smc[:,1] + (2*x_smc[:,2]))
plt.plot(y)
plt.show()




