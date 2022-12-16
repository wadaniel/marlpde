#!/bin/python3

# Discretization grid
N = 1024
N2 = 32

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from plotting import *
from Burger import *

from scipy.stats import gaussian_kde

def plotError(kx, rel_errors):
    figName = "relerror_quantiles.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))

    qs = np.quantile(rel_errors, [0.1, 0.5, 0.8], axis=0)
    ax.plot(kx, qs[1], color='coral') # median
    ax.fill_between(kx, qs[0], qs[2], alpha=0.2, color='coral')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-2,1e2])
    plt.tight_layout()
    plt.savefig(figName)
    plt.close()

def plotField(x,t, models):
    figName = "evolution2.pdf"
    print(f"[plotting] Plotting {figName} ...")
    print(len(models))
    
    fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
    colors = ['royalblue','coral']
    alphas = [1., 0.8] 
    for idx, model in enumerate(models):
        print(model)
        tEnd = 5
        dt = 0.001

        for i in range(16):
            t = i * tEnd / 16
            tidx = int(t/dt)
            k = int(i / 4)
            l = i % 4
            
            axs[k,l].plot(x[idx], model[tidx,:], '-', color=colors[idx], alpha=alphas[idx])

    fig.tight_layout()
    fig.savefig(figName)
    plt.close()

def plotSgsHistory(x, sgsHistory):

    sgsHistory = sgsHistory.flatten()
    sgsHistory = sgsHistory[np.abs(sgsHistory)<10]

    figName = "sgsHistory.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
  
    smax = sgsHistory.max()
    smin = sgsHistory.min()
    slevels = np.linspace(smin, smax, 50)
    svals = np.linspace(smin, smax, 500)
   
    sgsDensity = gaussian_kde(sgsHistory)
    #sgsDensityVals = sgsDensity(svals)
    #ax.plot(svals, sgsDensityVals)

    sfac = 3
    sgsMean = np.mean(sgsHistory)
    sgsSdev = np.std(sgsHistory)
    print(smin,sgsMean,smax)
    print(sgsSdev)
    svals  = np.linspace(sgsMean-sfac*sgsSdev,sgsMean+sfac*sgsSdev,500)
    ax.plot(svals, sgsDensity(svals))

    plt.tight_layout()
    fig.savefig(figName)
    plt.close('all')


def plotSgsField(x,t,sgsField):

    figName = "sgsField.pdf"
    print(f"[plotting] Plotting {figName} ...")
    print(sgsField.shape)
    print(np.max(sgsField))
    print(np.min(sgsField))
    
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    X, Y = np.meshgrid(x, t[:-1])
    print(x)
    print(t)
    ax.contourf(x, t, sgsField)
    plt.axis('equal')
    #ax.set_aspect('equal')
    #plt.tight_layout()
    plt.savefig(figName)
    plt.close('all')
    return

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L       = 2*pi
dt      = 0.001
s       = 20
tEnd    = 5
ic      = 'turbulence'
forcing = False
dsm=True#False
ssm=False

#L       = 100
#dt      = 0.01
#s       = 20
#tEnd    = 5000
#ic      = 'sinus'
#forcing = True

nu      = 0.02
#ic      = 'zero'
#ic      = 'turbulence'
#ic      = 'forced'
noise   = 0.
seed    = 42
#forcing = False

plotid = 2
load = True
if load == True:

    #np.savez(fname, dns_Ektt=dns_Ektt, sgs_Ektt=sgs_Ektt, sgs_actions=sgs_actions, sgs_u=sgs_u, dns_u=dns_u, indeces=indeces)
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)
    sgs = Burger(L=L, N=N2, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s, ssm=ssm, dsm=dsm)
    data = np.load('/scratch/wadaniel/episodes_303.npz')
    #data = np.load('/scratch/wadaniel/episodes_marl_303.npz')
    dns_Ektt = data['dns_Ektt']
    sgs_Ektt = data['sgs_Ektt']
    sgs_u = data['sgs_u']
    dns_u = data['dns_u']
    sgsHistory = data['sgs_actions']
    indeces = data['indeces']
    sgsIdx = np.where(indeces == plotid)[0]
    sgsIdx = sgsIdx[0]

    print(indeces)
    print(sgsIdx)
    print(dns_Ektt.shape)
    print(sgs_Ektt.shape)
    print(sgsHistory.shape)
    rel_errors = np.empty((0,N2//2-1), float) 
    for i in range(5):
        idx = (i+1)*5001
        rel_errors = np.vstack( (rel_errors, abs(dns_Ektt[idx-1, 1:N2//2]-sgs_Ektt[idx-1, 1:N2//2])/dns_Ektt[idx-1, 1:N2//2]) )

    #print(rel_errors)
    sgsField = sgsHistory[(sgsIdx*5001):(sgsIdx+1)*5001-1, :]
    uField = sgs_u[(sgsIdx*5001):(sgsIdx+1)*5001-1, :]
    dnsField = dns_u[(sgsIdx*5001):(sgsIdx+1)*5001-1, :]

else:

    dnsek_tt = np.empty((0,N//2), float)
    sgsek_tt = np.empty((0,N2//2), float)
    rel_errors = np.empty((0,N2//2-1), float)
    sgsHistory = np.empty((0,N2), float)
    for i in range(20):
        dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed+i, forcing=forcing, s=s)
        dns.simulate()
        dns.compute_Ek()
                
        v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:])) *  N2 / dns.N
     
        sgs = Burger(L=L, N=N2, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed+i, forcing=forcing, s=s, ssm=ssm, dsm=dsm)
        sgs.IC( v0 = v0 )
        sgs.simulate()
        sgs.compute_Ek()

        dns_e = dns.Ek_ktt[-1,:N//2]
        sgs_e = sgs.Ek_ktt[-1,:N2//2]
        rel_err = np.abs(sgs_e[1:]-dns.Ek_ktt[-1,1:N2//2])/dns.Ek_ktt[-1,1:N2//2]
        reward = np.mean(rel_err**2)
        print(reward)

        dnsek_tt = np.vstack( (dnsek_tt, dns.Ek_ktt[-1,:N//2]) )
        sgsek_tt = np.vstack( (sgsek_tt, sgs.Ek_ktt[-1,:N2//2]) )
        rel_errors = np.vstack( (rel_errors, rel_err) )

        if reward < 100:
            sgsHistory = np.vstack( ( sgsHistory, sgs.sgsHistory[:-1,:]) )
        print(sgsHistory)

        if (i == plotid):
            sgsField = sgs.sgsHistory[:-1,:]
            uField = sgs.uu
            dnsField = dns.uu

#plotError(sgs.k[1:N2//2], rel_errors)
#plotSgsHistory(sgs.x, sgsHistory)
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed+plotid, forcing=forcing, s=s)
dns.simulate()
plotField([dns.x, sgs.x], np.arange(5000)/1000, [dnsField, uField]) #uField])
plotSgsField(sgs.x, np.arange(5000)/1000, sgsField)
