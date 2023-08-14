import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns
import json

import numpy as np
from scipy import interpolate
from scipy.stats import gaussian_kde

colors = ['royalblue','coral']

def plotEvolution(model):
    figName = "evolution.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    alphas = [1., 0.8] 
    nsteps = model.nsteps

    fig, axs = plt.subplots(2,3, sharex=True, sharey=True) #, figsize=(15,15))
    for i in range(6):
        tidx = int(i * nsteps / 6)
        k = int(i / 3)
        l = i % 3
        
        axs[k,l].plot(model.x, model.uu[tidx,:], '-', color='royalblue', alpha=1.)
        axs[k,l].plot(model.x, model.gradientHistory[tidx,:], '-', color='coral', alpha=1.)
        axs[k,l].set_xticks([0, np.pi, 2*np.pi])

    fig.tight_layout()
    fig.savefig(figName)
    plt.close()

def plotActionField(model):
    figName = "actions.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    actions0 = model.actionHistory0
    actions1 = model.actionHistory1
    actions2 = model.actionHistory2

    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(6,6))
    axs[0].contourf(model.x, model.tt, actions0) #, ulevels)
    axs[0].set_xticks([0, np.pi, 2*np.pi])
    axs[1].contourf(model.x, model.tt, actions1) #, ulevels)
    axs[1].set_xticks([0, np.pi, 2*np.pi])
    contour = axs[2].contourf(model.x, model.tt, actions2) #, ulevels)
    plt.colorbar(contour)
    axs[2].set_xticks([0, np.pi, 2*np.pi])
    
    plt.tight_layout()

    plt.savefig(figName)
    plt.close()

def plotGradientField(model):
    figName = "grad.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    grad = model.gradientHistory

    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(10,10))
    contour = ax.contourf(model.x, model.tt, grad, levels=50)
    ax.set_xticks([0, np.pi, 2*np.pi])
    plt.colorbar(contour)
    
    plt.tight_layout()
    plt.savefig(figName)
    plt.close()

def plotActionDistribution(models):
    figName = "actiondist.pdf"
    print(f"[plotting] Plotting {figName} ...")
    actions0 = models.actionHistory0.flatten()
    actions1 = models.actionHistory1.flatten()
    actions2 = models.actionHistory2.flatten()
    #plt.violinplot(dataset=[actions])
    sns.violinplot(data=[actions0, actions1, actions2])
    plt.tight_layout()
    plt.savefig(figName)
    plt.close()
    return
