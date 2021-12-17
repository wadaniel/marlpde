#!/bin/python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../_model/')

from scipy import interpolate
import numpy as np
from KS import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 22/(2*np.pi)
N    = 512
dt   = 0.1
tEnd = 50
dns = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd)

## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

## for plotting
uTruth = dns.uu
tTruth = dns.tt
xTruth = dns.x
sTruth, nTruth = np.meshgrid(np.arange(uTruth.shape[0])*dt, 2*np.pi*L/N*(np.array(range(N))+1))

#------------------------------------------------------------------------------
## restart
u_restart = dns.uu[0,:].copy()
f_restart = interpolate.interp1d(xTruth, u_restart)

# restart from coarse physical space
N = 16
subgrid = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd)

subgrid.setGroundTruth(tTruth, xTruth, uTruth)

# create interpolated IC
xCoarse = subgrid.x
uRestartCoarse = f_restart(xCoarse)
print(uRestartCoarse)
subgrid.IC( u0 = uRestartCoarse)

# MARL LOOP
rewardHist = np.zeros((int(tEnd/dt)))
actions = np.ones(N)
for i in range(int(tEnd/dt)):
    state = subgrid.getState()
    #print("State")
    #print(state)
    subgrid.step()
    subgrid.updateField(actions)
    rewards = subgrid.getReward()
    rewardHist[i] = sum(rewards)
    #print("Reward")
    #print(rewards)

# convert to physical space
subgrid.fou2real()

# get solution
uCoarse = subgrid.uu

# eval truth on coarse Grid
uTruthToCoarse = subgrid.mapGroundTruth()

cumRewardHist = np.cumsum(rewardHist)
print("Cum Total Reward")
print(cumRewardHist)

#------------------------------------------------------------------------------
## plot comparison
fig, axs = plt.subplots(1,3)
s, n = np.meshgrid(subgrid.tt, subgrid.x)

cs0 = axs[0].contourf(sTruth, nTruth, uTruth.T, 50, cmap=plt.get_cmap("seismic"))
cs1 = axs[1].contourf(s, n, uCoarse.T, 50, cmap=plt.get_cmap("seismic"))
diff = np.abs(uCoarse-uTruthToCoarse)
print(diff)
cs2 = axs[2].contourf(s, n, diff.T, 50, cmap=plt.get_cmap("seismic"))

# plt.colorbar(cs0, ax=axs[0])
plt.colorbar(cs1, ax=axs[1])
plt.colorbar(cs2, ax=axs[2])
plt.setp(axs[:], xlabel='$t$')
plt.setp(axs[0], ylabel='$x$')
# for c in cs.collections: c.set_rasterized(True)
axs[1].set_yticklabels([])
axs[2].set_yticklabels([])
fig.savefig('marl.png')
