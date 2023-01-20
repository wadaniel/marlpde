import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns

import numpy as np
from scipy import interpolate
from scipy.stats import gaussian_kde

colors = ['royalblue','coral']

def plotEvolution(model):
    figName = "evolution.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    alphas = [1., 0.8] 
    tEnd = model.tend
    dt = model.dt

    fig, axs = plt.subplots(2,3, sharex=True, sharey=True) #, figsize=(15,15))
    for i in range(6):
        t = i * tEnd / 6
        tidx = int(t/dt)
        k = int(i / 3)
        l = i % 3
        
        axs[k,l].plot(model.x, model.uu[tidx,:], '-', color='royalblue', alpha=1.)
        axs[k,l].plot(model.x, model.solution[tidx,:], '--', color='coral', alpha=1.)

    fig.tight_layout()
    fig.savefig(figName)
    plt.close()

def plotError(model):
    figName = "abserror.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    abserror = np.abs(model.uu - model.solution)

    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
    contour = ax.contourf(model.x, model.tt, abserror) #, ulevels)
    plt.colorbar(contour)
    
    plt.tight_layout()
    plt.savefig(figName)
    plt.close()

    ####
    figName = "relerror.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    relerror = np.abs(model.uu - model.solution)/np.max(np.abs(model.solution), axis=1)[:,np.newaxis]

    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
    contour = ax.contourf(model.x, model.tt, relerror) #, ulevels)
    plt.colorbar(contour)
    
    plt.tight_layout()
    plt.savefig(figName)
    plt.close()


def plotActionField(model):
    figName = "actionfield.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    actions = model.actionHistory

    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
    contour = ax.contourf(model.x, model.tt, actions) #, ulevels)
    plt.colorbar(contour)
    
    plt.tight_layout()
    plt.savefig(figName)
    plt.close()

def plotDiffusionField(model):
    figName = "diffusionfield.pdf"
    print(f"[plotting] Plotting {figName} ...")
    
    actions = model.actionHistory

    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(10,10))
    ax.contourf(model.x, model.tt, actions) #, ulevels)
    
    plt.tight_layout()
    plt.savefig(figName)
    plt.close()

def plotActionDistribution(models):
    figName = "actiondist.pdf"
    print(f"[plotting] Plotting {figName} ...")
    actions = models.actionHistory.flatten()
    #plt.violinplot(dataset=[actions])
    sns.violinplot(data=[actions])
    plt.tight_layout()
    plt.savefig(figName)
    plt.close()
    return
