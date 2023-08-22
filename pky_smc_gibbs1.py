from pstats import Stats
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from pky_simulator import PKYSim
from torch.distributions import Uniform
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abcsmc.abcsmc import ABCSMC
from util.mmd import GaussianKernel
from torch.distributions import Uniform, Beta, HalfNormal, Gamma
import time as timer
from CPP import pkyssa
import pickle
import torch.distributions as dist
import torch
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

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

if __name__ == '__main__':

    import pandas as pd
    import numpy as np


    ppc_abc_lst =[]
    ppc_le_lst =[]
    ppc_re_lst =[]
    inc_ppc_le_lst = []
    inc_ppc_re_lst = []
    smc_le_lst = []
    smc_re_lst = []

    dist_abc_lst = []
    dist_ppc_le_lst = []
    dist_ppc_re_lst = []
    dist_inc_ppc_le_lst = []
    dist_inc_ppc_re_lst = []
    dist_smc_le_lst = []
    dist_smc_re_lst = []

    nll_abc_lst = []
    nll_le_lst = []
    nll_re_lst = []

    variances = np.zeros((7,10))
    distances = np.zeros((7,10))
    negLik = np.zeros((3,10))

    mmds = np.zeros((6,10))
    mmd_idele_lst = []
    mmd_prdynle_lst = []
    mmd_abcle_lst = []
    mmd_idere_lst = []
    mmd_prdynre_lst = []
    mmd_abcre_lst = []     

    time = 100
    num_sim = 5000
    _prior = [Gamma(torch.tensor([2.]), torch.tensor([3.])) for _ in range(8)]
    num_dim = 8
    _simulator = PKYSim(simulate_summary=False)
    simulator, prior = prepare_for_sbi(_simulator, _prior)
    proposal = prior
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_sim)
    noisy_x = (x[:,:,1] + (2*x[:,:,2])) + (np.random.randn(num_sim,time).astype(np.float32)*2)
    
    species, ys, constants = x.detach().numpy(), noisy_x.detach().numpy().reshape((-1,time,1)), theta.detach().numpy()
    species = species + np.random.rand(*species.shape)

    Path=np.concatenate((species,np.roll(ys,-1,axis=1),constants.reshape((num_sim,1,8)).repeat(time,axis=1)),axis=2)
    path = np.array(Path, dtype=np.float32)
    x_next = np.concatenate(np.split(path[:num_sim,1::2,:4],int(time/2),axis=1),axis=0).squeeze()
    x_prev = np.concatenate(np.split(path[:num_sim,0::2,:],int(time/2),axis=1),axis=0).squeeze() 

    density_estimator_build_fun = posterior_nn(
        model="maf", hidden_features=50, num_transforms=5
    )
    infer = SNPE(density_estimator=density_estimator_build_fun)#, device="cuda")
    x_next_ = torch.from_numpy(x_next)#.to(device)
    x_prev_ = torch.from_numpy(x_prev)#.to(device)
    inference = infer.append_simulations(x_next_, x_prev_)
    density_estimator= inference.train(training_batch_size=256)

    Path_gibbs=np.concatenate((species,np.roll(species,-2,axis=1),np.roll(ys,-1,axis=1),\
        constants.reshape((num_sim,1,8)).repeat(time,axis=1)),axis=2)
    path_gibbs = np.array(Path_gibbs[:,:-2,:], dtype=np.float32)
    x_next = np.concatenate(np.split(path_gibbs[:,1::4,:4],int(time/4),axis=1),axis=0).squeeze()
    x_prev = np.concatenate(np.split(path_gibbs[:,0::4,:],int(time/4),axis=1),axis=0).squeeze()  
    density_estimator_build_fun = posterior_nn(
        model="maf", hidden_features=50, num_transforms=5
    )
    infer_gibbs = SNPE(density_estimator=density_estimator_build_fun)#, device="cuda")
    x_next_gibbs = torch.from_numpy(x_next)#.to(device)
    x_prev_gibbs = torch.from_numpy(x_prev)#.to(device)
    inference_gibbs = infer_gibbs.append_simulations(x_next_gibbs, x_prev_gibbs)
    density_estimator_gibbs=inference_gibbs.train(training_batch_size=256)
    for id in range(1,3):

        id_number = id
        times = np.arange(0,50,.5)
        gen_par = np.array([0.1,0.7,0.35,0.2,0.1,0.9,0.3,0.1])
        sigma = 2
        samplesize = 500
        param_filename = './results/data/'+str(id_number)+'pkysum.p'
        y = pickle.load(open(param_filename, 'rb'))
        y = y.reshape((-1,1))
        print(y.shape)
        param_filename = './results/paper/pky_path/pky_data.p'
        pickle.dump(y, open(param_filename, 'wb')) 
        y_rep1 = np.array([y]*500)

        param_filename = './results/params/'+str(id_number)+'pkyre_nosum_.p'
        params_re = pickle.load(open(param_filename, 'rb'))  

        T=y.shape[0]
        P=5000
        y_new = y
        y_rep=np.array([y_new]*P)
        inc_ppc_re = []
        d = 4
        for i in range(params_re.shape[0]):

            theta_rep = np.array([params_re[i,:]]*P).reshape((P,8))
            x_ = np.zeros((P, T,d))
            x_[:,0,:] = np.zeros(d)
            for n in range(1,T):
                cond = torch.from_numpy(np.hstack((x_[:,n-1,:],y_rep[:,n,:],theta_rep)).astype(np.float32)).view((-1,13))
                x_[:,n,:] = density_estimator.sample(1,cond).detach().squeeze()

            x_tch = torch.from_numpy(x_.astype(np.float32))
            w = np.zeros((T,P))
            x__=x_.copy()
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
            print('done ',i)
            inc_ppc_re.append(np.floor(x_p[0]))
        inc_ppc_re = np.array(inc_ppc_re)
        inc_ppc_re = inc_ppc_re[...,1] + 2*inc_ppc_re[...,2] 
        inc_ppc_re += np.random.randn(*inc_ppc_re.shape)*sigma
        param_filename = './results/params/'+str(id_number)+'pkyle.p' 
        params_le = pickle.load(open(param_filename, 'rb'))  
        param_filename = './results/ppc/'+str(id_number)+'pkyle_ppc.p'
        inc_ppc_le = pickle.load(open(param_filename, 'rb'))   
        inc_ppc_le = inc_ppc_le[...,1] + 2*inc_ppc_le[...,2] 
        inc_ppc_le += np.random.randn(*inc_ppc_le.shape)*sigma


        param_filename = './results/abc_tol/params/'+str(id_number)+'pkyabc_tol_nosum.p' 
        params_abc = pickle.load(open(param_filename, 'rb')) 
        param_filename = './results/abc_tol/ws/'+str(id_number)+'pky_w_tol_nosum.p' 
        params_abc_ws = pickle.load(open(param_filename, 'rb'))         
        param_filename = './results/abc_tol/xs/'+str(id_number)+'pky_x_tol_nosum.p' 
        ppc_abc = pickle.load(open(param_filename, 'rb'))   
        ppc_abc = ppc_abc[...,1] + 2*ppc_abc[...,2] 
        ppc_abc += np.random.randn(*ppc_abc.shape)*sigma    
        


        """
        sns.set_context("paper", font_scale=1)
        sns.set(rc={"figure.figsize":(15,13),"font.size":19,"axes.titlesize":19,"axes.labelsize":19,
                "xtick.labelsize":18, "ytick.labelsize":18},style="white")
        param_names = [r"$c_1$",r"$c_2$",r"$c_3$",r"$c_4$",r"$c_5$",r"$c_6$",r"$c_7$",r"$c_8$"]
        real_params = gen_par

        sde_params_le = params_le
        sde_params_re = params_re  
        sde_params_abc = params_abc      
        for i, p in enumerate(param_names):

            
            # Add histogram subplot
            plt.subplot(3, 3, i+1)
            plt.axvline(real_params[i], linewidth=2.5, color='black')
            if i==0:
                sns.kdeplot(sde_params_le[:, i], color='red', linewidth = 2.5, label='SNLE')
                sns.kdeplot(sde_params_re[:, i], color='orange', linewidth = 2.5, label='SRE')
                sns.kdeplot(sde_params_abc[:, i], color='magenta', linewidth = 2.5, label='ABC')
            elif i==1:
                sns.kdeplot(sde_params_le[:, i], color='red', linewidth = 2.5, label='SNLE')
                sns.kdeplot(sde_params_re[:, i], color='orange', linewidth = 2.5, label='SRE')
                sns.kdeplot(sde_params_abc[:, i], color='magenta', linewidth = 2.5, label='ABC')
            else:
                sns.kdeplot(sde_params_le[:, i], color='red', linewidth = 2.5, label='SNLE')
                sns.kdeplot(sde_params_re[:, i], color='orange', linewidth = 2.5, label='SRE')
                sns.kdeplot(sde_params_abc[:, i], color='magenta', linewidth = 2.5, label='ABC')


            plt.xlabel(param_names[i])    
            plt.ylabel('Frequency')    
            if i==1:
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=3,fontsize=22)
        plt.subplots_adjust(hspace=0.7)
        plt.tight_layout()
        #plt.show()
        plt.savefig('./results/figures/pkypars'+str(id_number)+'.png',dpi=300)
        """
        
        simulator = PKYSim(False)
        ppc_re = simulator(params_re)

        if id==1:
            param_filename = './results/paper/pky_path/pky_ppc_re.p'
            pickle.dump(ppc_re, open(param_filename, 'wb'))   
        ppc_re = ppc_re[...,1] + 2*ppc_re[...,2]       
        ppc_re += np.random.randn(*ppc_re.shape)*sigma 
        ppc_le = simulator(params_le)
        
        if id==1:
            param_filename = './results/paper/pky_path/pky_ppc_le.p'
            pickle.dump(ppc_le, open(param_filename, 'wb')) 
        ppc_le = ppc_le[...,1] + 2*ppc_le[...,2]             
        ppc_le += np.random.randn(*ppc_le.shape)*sigma 

        

        smc_le = np.zeros((samplesize, len(times),4))
        smc_re = np.zeros((samplesize, len(times),4))
        out = np.array(Parallel(n_jobs=8)(delayed(BootStrapp)(y, np.array(params_re[i,:]), 1000, times) for i in range(samplesize)))
        for i in range(samplesize):
            #smc_le[i,:] = BootStrapp(y, np.array(params_le[i,:]), 100, times)
            smc_re[i,:] = BootStrapp(y, np.array(params_re[i,:]), 100, times)
        if id==1:
            param_filename = './results/paper/pky_path/pky_smc_re.p'
            pickle.dump(smc_re, open(param_filename, 'wb'))
           
               

        smc_le = smc_le[...,1] + 2*smc_le[...,2] 
        smc_le += np.random.randn(*smc_le.shape)*sigma 
        smc_re = smc_re[...,1] + 2*smc_re[...,2] 
        smc_re += np.random.randn(*smc_re.shape)*sigma 
        

        inc_ppc_le = inc_ppc_le.reshape((500,100,1))
        inc_ppc_re = inc_ppc_re.reshape((500,100,1))
        ppc_le = ppc_le.reshape((500,100,1))
        ppc_re = ppc_re.reshape((500,100,1))
        smc_le = smc_le.reshape((500,100,1))
        smc_re = smc_re.reshape((500,100,1))
        ppc_abc = ppc_abc.reshape((500,100,1))

        
        kern=GaussianKernel(1.5)
        sig=kern.get_sigma_median_heuristic(smc_le[:,50,:],500)
        kernmd=GaussianKernel(sig)
        
        
        mmd_idele=np.sum([kernmd.estimateMMD(smc_le[:,i,:],inc_ppc_le[:,i,:],True) for i in range(100)])
        mmd_prdynle=np.sum([kernmd.estimateMMD(smc_le[:,i,:],ppc_le[:,i,:],True) for i in range(100)])
        mmd_abcle=np.sum([kernmd.estimateMMD(smc_le[:,i,:],ppc_abc[:,i,:],True) for i in range(100)])

        mmd_idere=np.sum([kernmd.estimateMMD(smc_re[:,i,:],inc_ppc_re[:,i,:],True) for i in range(100)])
        mmd_prdynre=np.sum([kernmd.estimateMMD(smc_re[:,i,:],ppc_re[:,i,:],True) for i in range(100)])
        mmd_abcre=np.sum([kernmd.estimateMMD(smc_re[:,i,:],ppc_abc[:,i,:],True) for i in range(100)])

        mmd_idele_lst.append(mmd_idele)
        mmd_prdynle_lst.append(mmd_prdynle)
        mmd_abcle_lst.append(mmd_abcle)
        mmd_idere_lst.append(mmd_idere)
        mmd_prdynre_lst.append(mmd_prdynre)
        mmd_abcre_lst.append(mmd_abcre)
        

        """
        ppc_abc_lst.append(np.var(ppc_abc,axis=0).sum())
        ppc_le_lst.append(np.var(ppc_le,axis=0).sum())
        ppc_re_lst.append(np.var(ppc_re,axis=0).sum())
        inc_ppc_le_lst.append(np.var(inc_ppc_le,axis=0).sum())
        inc_ppc_re_lst.append(np.var(inc_ppc_re,axis=0).sum())
        smc_le_lst.append(np.var(smc_le,axis=0).sum())
        smc_re_lst.append(np.var(smc_re,axis=0).sum())
        

        dist_abc = np.linalg.norm(y_rep1-ppc_abc,axis=(1,2))**2
        dist_ppc_le = np.linalg.norm(y_rep1-ppc_le,axis=(1,2))**2
        dist_ppc_re = np.linalg.norm(y_rep1-ppc_re,axis=(1,2))**2
        dist_inc_ppc_le = np.linalg.norm(y_rep1-inc_ppc_le,axis=(1,2))**2
        dist_inc_ppc_re = np.linalg.norm(y_rep1-inc_ppc_re,axis=(1,2))**2
        dist_smc_le = np.linalg.norm(y_rep1-smc_le,axis=(1,2))**2
        dist_smc_re = np.linalg.norm(y_rep1-smc_re,axis=(1,2))**2

        dist_abc_lst.append(dist_abc.mean())
        dist_ppc_le_lst.append(dist_ppc_le.mean())
        dist_ppc_re_lst.append(dist_ppc_re.mean())
        dist_inc_ppc_le_lst.append(dist_inc_ppc_le.mean())
        dist_inc_ppc_re_lst.append(dist_inc_ppc_re.mean())
        dist_smc_le_lst.append(dist_smc_le.mean())
        dist_smc_re_lst.append(dist_smc_re.mean())    
        """
        ind = np.random.choice(500,size=(500,),p=params_abc_ws)
        particles = params_abc#[ind,:]
        gm_abc = GaussianMixture(n_components=3, random_state=0).fit(particles)       
        nll_abc_lst.append(gm_abc.score_samples(gen_par.reshape((1,-1))).squeeze())
        gm_le = GaussianMixture(n_components=3, random_state=0).fit(params_le)       
        nll_le_lst.append(gm_le.score_samples(gen_par.reshape((1,-1))).squeeze())
        gm_re = GaussianMixture(n_components=3, random_state=0).fit(params_re)       
        nll_re_lst.append(gm_re.score_samples(gen_par.reshape((1,-1))).squeeze())

        #nll_abc_lst.append(stats.multivariate_normal(mean=np.mean(params_abc,axis=0),cov=np.cov(params_abc.T)).logpdf(gen_par)) 
        #nll_le_lst.append(stats.multivariate_normal(mean=np.mean(params_le,axis=0),cov=np.cov(params_le.T)).logpdf(gen_par))
        #nll_re_lst.append(stats.multivariate_normal(mean=np.mean(params_re,axis=0),cov=np.cov(params_re.T)).logpdf(gen_par))            
    
    """
    variances[0,:] = np.array(ppc_abc_lst)
    variances[1,:] = np.array(ppc_le_lst)
    variances[2,:] = np.array(ppc_re_lst)
    variances[3,:] = np.array(smc_le_lst)
    variances[4,:] = np.array(smc_re_lst)
    variances[5,:] = np.array(inc_ppc_le_lst)
    variances[6,:] = np.array(inc_ppc_re_lst)
    
    mmds[0,:] = np.array(mmd_idele_lst)
    mmds[1,:] = np.array(mmd_prdynle_lst)
    mmds[2,:] = np.array(mmd_abcle_lst)
    mmds[3,:] = np.array(mmd_idere_lst)
    mmds[4,:] = np.array(mmd_prdynre_lst)
    mmds[5,:] = np.array(mmd_abcre_lst)    
    np.savetxt('./results/paper/pky_mmds',mmds)
   

    distances[0,:] = np.array(dist_abc_lst)
    distances[1,:] = np.array(dist_ppc_le_lst)
    distances[2,:] = np.array(dist_ppc_re_lst)
    distances[3,:] = np.array(dist_smc_le_lst)
    distances[4,:] = np.array(dist_smc_re_lst)
    distances[5,:] = np.array(dist_inc_ppc_le_lst)
    distances[6,:] = np.array(dist_inc_ppc_re_lst)
     """
    print('mean',np.mean(mmd_abcle_lst))
    print('sd',np.std(mmd_abcle_lst))
    negLik[0,:] = np.array(nll_abc_lst)
    negLik[1,:] = np.array(nll_le_lst)
    negLik[2,:] = np.array(nll_re_lst)
    print(negLik.mean(axis=1))
    np.savetxt('./results/paper/pky_negLik_nosum.txt',negLik)
        
    mmds[0,:] = np.array(mmd_idele_lst)
    mmds[1,:] = np.array(mmd_prdynle_lst)
    mmds[2,:] = np.array(mmd_abcle_lst)
    mmds[3,:] = np.array(mmd_idere_lst)
    mmds[4,:] = np.array(mmd_prdynre_lst)
    mmds[5,:] = np.array(mmd_abcre_lst)    
    np.savetxt('./results/paper/pky_mmds_nosum.txt',mmds)