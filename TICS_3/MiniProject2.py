
#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
@author: created by David Munoz
"""

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy.stats import norm
import cma
import seaborn as sns
import time
import os
homedir = os.path.expanduser('~/Desktop/TrendsCN/tics-3/')

df = pd.read_csv('./data/KS014_train.csv')              # Load .csv file into a pandas DataFrame

df['signed_contrast'] = df['contrast']*df['position']   # We define a new column for "signed contrasts"
df.drop(columns='stim_probability_left', inplace=True)  # Stimulus probability has no meaning for training sessions

print('Total # of trials: ' + str(len(df['trial_num'])))
print('Sessions: ' + str(np.unique(df['session_num'])))
#df.head()

def psychofun(theta,stim):
    """Psychometric function based on normal CDF and lapses"""
    mu = theta[0]          # bias
    sigma = theta[1]       # slope/noise
    lapse = theta[2]       # lapse rate
    if len(theta) == 4:    # lapse bias
        lapse_bias = theta[3]
    else:
        lapse_bias = 0.5   # if theta has only three elements, assume symmetric lapses
    
    p_right = norm.cdf(stim,loc=mu,scale=sigma)    # Probability of responding "rightwards", without lapses
    p_right = lapse*lapse_bias + (1-lapse)*p_right # Adding lapses

    return p_right

# def psychofun_plot(theta,ax):
#     """Plot psychometric function"""    
#     stim = np.linspace(-100,100,201)   # Create stimulus grid for plotting    
#     p_right = psychofun(theta,stim)    # Compute psychometric function values
#     ax.plot(stim,p_right,label='model')
#     ax.legend()
#     return

def psychofun_loglike(theta,df):
    """Log-likelihood for psychometric function model"""
    s_vec = df['signed_contrast'] # Stimulus values
    r_vec =  df['response_choice']  # Responses
    
    p_right = psychofun(theta,s_vec)
    
    # Compute summed log likelihood for all rightwards and leftwards responses
    loglike = np.sum(np.log(p_right[r_vec == 1])) + np.sum(np.log(1 - p_right[r_vec == -1]))

    return loglike

# Define hard parameter bounds
lb = np.array([-100,1,0,0])
ub = np.array([100,100,1,1])
bounds = [lb,ub]

# Define plausible range
plb = np.array([-25,5,0.05,0.2])
pub = np.array([25,25,0.40,0.8])

theta1 = []
theta2 = []
theta3 = []
theta4 = [] 
negLL =  []

for session_num in range(1,16): # loop over session data

    df_session = df[df['session_num'] == session_num]
    # Define objective function: negative log-likelihood
    opt_fun = lambda theta_: -psychofun_loglike(theta_,df_session)
    
    # Generate random starting point for the optimization inside the plausible box
    theta0 = np.random.uniform(low=plb,high=pub) 

    # Initialize CMA-ES algorithm
    opts = cma.CMAOptions()
    opts.set("bounds",bounds)
    opts.set("tolfun",1e-5)

    # Run optimization
    res = cma.fmin(opt_fun, theta0, 0.5, opts)

    print('')
    print('Session') 
    print(session_num)
    print('Returned parameter vector: ' + str(res[0]))
    print('Negative log-likelihood at solution: ' + str(res[1]))
    vec =res[0]

    theta1.append(vec[0])
    theta2.append(vec[1])
    theta3.append(vec[2])
    theta4.append(vec[3])
    
    negLL.append(res[1])

    #fig = plt.figure(figsize=(9,4))
    #ax = plot_psychometric_data(df_session,session_num)
    #psychofun_plot(res[0],ax)
    #plt.show()

#1a)

session = np.linspace(1,15,15, dtype=int)

def plot_sess_param(X, Y1, label1, Y2=None,  label2=None, labelALL=None, caption=None):  
    sns.set_style("darkgrid")
    plt.figure(figsize=(9, 6))  
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    #plt.ylim(63, 85)  
    plt.xticks(range(1, 16, 1), fontsize=10)  
    #plt.yticks(range(65, 86, 5), fontsize=14) 
    plt.xlabel("Session", fontsize=16)
    
    if caption!=None:
        plt.figtext(0.5, 0.9, caption, wrap=False, horizontalalignment='center', fontsize=12)
        
    if Y2==None:
        plt.ylabel(label1, fontsize=16) 
        plt.plot(X, Y1, color="blue", lw=2)  
    else:
        plt.plot(X, Y1, color="blue", lw=2, label=label1)  
        plt.plot(X, Y2, color="orange", lw=2, label=label2)  
        plt.ylabel(labelALL, fontsize=16) 
        plt.legend()
    #plt.title("Theta as a function of session", fontsize=22)  

plot_sess_param(session, theta1, 'Bias $\mu$')
plt.savefig(homedir+"Bias.png")

plot_sess_param(session, theta2, 'Slope $\sigma$')
plt.savefig(homedir+"Slope.png")

plot_sess_param(session, theta3, 'Lapse Rate')
plt.savefig(homedir+"LapseR.png")

plot_sess_param(session, theta4, 'Lapse Bias')
plt.savefig(homedir+"LapseB.png")

#1b)

def psychofun_repeatlast_loglike(theta,df):
    """Log-likelihood for last-choice dependent psychometric function model"""
    s_vec = np.array(df['signed_contrast']) # Stimulus values
    r_vec = np.array(df['response_choice'])  # Responses
    
    p_last = theta[0] # Probability of responding as last choice
    theta_psy = theta[1:] # Standard psychometric function parameters
        
    p_right = psychofun(theta_psy,s_vec)
    
    # Starting from the 2nd trial, probability of responding equal to the last trial
    p_right[1:] = p_last*(r_vec[0:-1] == 1) + (1-p_last)*p_right[1:] 
    
    # Compute summed log likelihood for all rightwards and leftwards responses
    loglike = np.sum(np.log(p_right[r_vec == 1])) + np.sum(np.log(1 - p_right[r_vec == -1]))

    return loglike


lb = np.array([0,-100,1,0,0])
ub = np.array([1,100,100,1,1])
bounds = [lb,ub]

plb = np.array([0.05,-25,5,0.05,0.2])
pub = np.array([0.2,25,25,0.45,0.8])
 
Rep_negLL =  []
ntrials =  []


for session_num in range(1,16): # loop over session data

    df_session = df[df['session_num'] == session_num]
    trial_mask = df['session_num'] == session_num # Indexes of trials of the chosen session
    ntrials.append(np.sum(trial_mask))

    # df_session = df[(df['session_num'] == session_num) & (df['trial_num'] > 300)]
    opt_fun = lambda theta_: -psychofun_repeatlast_loglike(theta_,df_session)

    theta0 = np.random.uniform(low=plb,high=pub)
    opts = cma.CMAOptions()
    opts.set("bounds",bounds)
    opts.set("tolfun",1e-5)
    res_repeatlast = cma.fmin(opt_fun, theta0, 0.5, opts)

    print('')
    print('Session') 
    print(session_num)
    print('Returned parameter vector: ' + str(res_repeatlast[0]))
    print('Negative log-likelihood at solution: ' + str(res_repeatlast[1]))
    
    Rep_negLL.append(res_repeatlast[1])
    
    #fig = plt.figure(figsize=(9,4))
    #ax = plot_psychometric_data(df_session,session_num)
    #psychofun_plot(res[0],ax)
    #plt.show()

Nmodels = 2
nparams = np.zeros(Nmodels)
results = [res,res_repeatlast] # Store all optimization output in a vector
for i in range(0,len(results)):
    nparams[i] = len(results[i][0])

nll = np.column_stack((negLL,Rep_negLL))

deltaAIC = []
deltaBIC = []


#model selection
for k in range(0,15): # loop over session data
    session_num = k+1
    aic = []
    bic = []
    for i in range(0,2):
        aic.append(2*nll[k,i] + 2*nparams[i])
        #print(nparams[i])
        bic.append(2*nll[k,i] + nparams[i]*np.log(ntrials[k]))
    diff =2*nll[k,1] - 2*nll[k,0]
    #print(diff)
    pen = nparams[1]*np.log(ntrials[k]) - 2*nparams[0]
    #print(diff + pen)
    
    deltaAIC.append(aic[0] - aic[1])
    deltaBIC.append(bic[0] - bic[1])
    
    
    print('Model comparison results session ' + str(session_num))
    print('The bigger, the more favorable for Repeat model')
    #print('delta AIC (basic - Repeated): ' + str(aic[0] - aic[1]))
    #print('BIC model 1 ' + str(bic[0]))
    #print('BIC model 2 ' + str(bic[1]))
    diffB = bic[0] - bic[1]
    #print('AIC model 1 ' + str(aic[0]))
    #print('AIC model 2 ' + str(aic[1]))
    diffA = aic[0] - aic[1]
    print('delta BIC (basic - Repeated) : ' + str(diffB))
    print('delta AIC (basic - Repeated) : ' + str(diffA))
    #print('delta : ' + str(deltaAIC[k] - deltaBIC[k]))
    #print('diff ' + str(test - test2))
    print('')

plot_sess_param(session, deltaAIC, '$\Delta$ AIC',  deltaBIC, '$\Delta$ BIC', '$\Delta$ IC (model 1 - model 2)', 'Model 1 = Normal CDF+lapses, Model 2 = Repeat last-choice')
plt.savefig(homedir+"AIC_BIC.png")

diff = np.asarray(deltaAIC) - np.asarray(deltaBIC)
plot_sess_param(session, diff, '$\Delta$ Between ICs (AIC-BIC)')
plt.savefig(homedir+"IC_diff.png")



def psychofun_vec(theta,stim):
    """Psychometric function based on normal CDF and lapses"""
    mu = theta[:,0]          # bias
    sigma = theta[:,1]       # slope/noise
    lapse = theta[:,2]      # lapse rate
    if len(theta) == 4:    # lapse bias
        lapse_bias = theta[:,3]
    else:
        lapse_bias = np.linspace(0.5,0.5,len(theta))   # if theta has only three elements, assume symmetric lapses
    
    p_right = norm.cdf(stim,loc=mu,scale=sigma)    # Probability of responding "rightwards", without lapses
    p_right = lapse*lapse_bias + (1-lapse)*p_right # Adding lapses

    return p_right

def psychofun_timevarying_loglike_vec(theta,df):
    """Log-likelihood for time-varying psychometric function model"""
    s_vec = np.array(df['signed_contrast']) # Stimulus values
    r_vec = np.array(df['response_choice'])  # Responses

    Ntrials = len(s_vec)
    mu_vec = np.linspace(theta[0],theta[4],Ntrials)
    sigma_vec = np.linspace(theta[1],theta[5],Ntrials)
    lapse_vec = np.linspace(theta[2],theta[6],Ntrials)
    lapsebias_vec = np.linspace(theta[3],theta[7],Ntrials)
    ThetaVec = np.transpose(np.asarray([mu_vec,sigma_vec,lapse_vec,lapsebias_vec]))
    p_right = np.zeros(Ntrials)
    
    
    p_right = psychofun_vec(ThetaVec,s_vec)
    #print(p_right)
    # Compute summed log likelihood for all rightwards and leftwards responses
    loglike = np.sum(np.log(p_right[r_vec == 1])) + np.sum(np.log(1 - p_right[r_vec == -1]))

    return loglike

def psychofun_timevarying_loglike(theta,df):
    """Log-likelihood for time-varying psychometric function model"""
    s_vec = np.array(df['signed_contrast']) # Stimulus values
    r_vec = np.array(df['response_choice'])  # Responses

    Ntrials = len(s_vec)
    mu_vec = np.linspace(theta[0],theta[4],Ntrials)
    sigma_vec = np.linspace(theta[1],theta[5],Ntrials)
    lapse_vec = np.linspace(theta[2],theta[6],Ntrials)
    lapsebias_vec = np.linspace(theta[3],theta[7],Ntrials)

    p_right = np.zeros(Ntrials)
    
    for t in range(0,Ntrials):
        p_right[t] = psychofun([mu_vec[t],sigma_vec[t],lapse_vec[t],lapsebias_vec[t]],s_vec[t])
    #print(p_right)
    # Compute summed log likelihood for all rightwards and leftwards responses
    loglike = np.sum(np.log(p_right[r_vec == 1])) + np.sum(np.log(1 - p_right[r_vec == -1]))

    return loglike


lb = np.array([-100,1,0,0,-100,1,0,0])
ub = np.array([100,100,1,1,100,100,1,1])
bounds = [lb,ub]

plb = np.array([-25,5,0.05,0.2,-25,5,0.05,0.2])
pub = np.array([25,25,0.45,0.8,25,25,0.45,0.8])

#with for loops

TimeF_negLL =  []

start_time = time.time()

for session_num in range(1,16): # loop over session data

    df_session = df[df['session_num'] == session_num]

    opt_fun = lambda theta_: -psychofun_timevarying_loglike(theta_,df_session)

    theta0 = np.random.uniform(low=plb,high=pub)
    opts = cma.CMAOptions()
    opts.set("bounds",bounds)
    opts.set("tolfun",1e-5)
    res_time = cma.fmin(opt_fun, theta0, 0.5, opts)

    print('')
    print('Session') 
    print(session_num)
    print('Returned parameter vector: ' + str(res_time[0]))
    print('Negative log-likelihood at solution: ' + str(res_time[1]))
    
    TimeF_negLL.append(res_time[1])

    #fig = plt.figure(figsize=(9,4))
    #ax = plot_psychometric_data(df_session,session_num)
    #psychofun_plot(res[0],ax)
    #plt.show()

print("--- %s seconds with for loops ---" % (time.time() - start_time))


#vectorized

Time_negLL =  []

start_time = time.time()

for session_num in range(1,16): # loop over session data

    df_session = df[df['session_num'] == session_num]

    opt_fun = lambda theta_: -psychofun_timevarying_loglike_vec(theta_,df_session)

    theta0 = np.random.uniform(low=plb,high=pub)
    opts = cma.CMAOptions()
    opts.set("bounds",bounds)
    opts.set("tolfun",1e-5)
    res_time = cma.fmin(opt_fun, theta0, 0.5, opts)

    print('')
    print('Session') 
    print(session_num)
    print('Returned parameter vector: ' + str(res_time[0]))
    print('Negative log-likelihood at solution: ' + str(res_time[1]))
    
    Time_negLL.append(res_time[1])

    #fig = plt.figure(figsize=(9,4))
    #ax = plot_psychometric_data(df_session,session_num)
    #psychofun_plot(res[0],ax)
    #plt.show()
print("--- %s seconds ---" % (time.time() - start_time))


Nmodels = 3
nparams = np.zeros(Nmodels)
results = [res,res_repeatlast,res_time] # Store all optimization output in a vector
for i in range(0,len(results)):
    nparams[i] = len(results[i][0])

nll = np.column_stack((negLL,Rep_negLL, Time_negLL))

deltaAIC_rep = []
deltaBIC_rep = []

deltaAIC_time = []
deltaBIC_time = []

#model selection
for i in range(0,15): # loop over session data
    session_num = i+1
    aic = []
    bic = []
    for j in range(0,Nmodels):
        aic.append(2*nll[i,j] + 2*nparams[j])
        #print(nparams[i])
        bic.append(2*nll[i,j] + nparams[j]*np.log(ntrials[i]))
    #diff =2*nll[k,1] - 2*nll[k,0]
    #print(diff)
    #pen = nparams[1]*np.log(ntrials[k]) - 2*nparams[0]
    #print(diff + pen)
    deltaAIC = []
    deltaBIC = []
    for k in range(1,Nmodels):
        deltaAIC.append(aic[0] - aic[k])
        deltaBIC.append(bic[0] - bic[1])
        
    deltaAIC_rep.append(deltaAIC[0])
    deltaBIC_rep.append(deltaBIC[0])
    
    deltaAIC_time.append(deltaAIC[1])
    deltaBIC_time.append(deltaBIC[1])
    
    
    print('Model comparison results session ' + str(session_num))
    print('The bigger, the more favorable for the alternative model')
    print('delta BIC (basic - Repeated) : ' + str(deltaBIC[0]))
    print('delta AIC (basic - Repeated) : ' + str(deltaAIC[0]))
    print('delta BIC (basic - Time) : ' + str(deltaBIC[1]))
    print('delta AIC (basic - Time) : ' + str(deltaAIC[1]))
    print('')


plot_sess_param(session, deltaAIC_time, '$\Delta$ AIC',  deltaBIC_time, '$\Delta$ BIC', '$\Delta$ IC (model 1 - model 3)', 'Model 1 = Normal CDF+lapses, Model 3 = Time-varying')
plt.savefig(homedir+"AIC_BIC_time-varying.png")

diff = np.asarray(deltaAIC_time) - np.asarray(deltaBIC_time)
plot_sess_param(session, diff, '$\Delta$ Between ICs (AIC-BIC)')
plt.savefig(homedir+"IC_diff_time.png")