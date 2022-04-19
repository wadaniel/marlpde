#!/bin/python3

"""
This scripts simulates the KS on a fine grid (N1) and on a coare grid (N2)
The IC in the coarse grid is the interpolated IC of the fine grid. Then we plot 
the KS of the fine grid, the KS on the coarse grid, and the difference between 
KS of fine grid interpolated on coarse crid vs KS on coarse grid.

"""

# Discretization fine grid (DNS)
N1 = 2048

# Discretization coarse grid
N2 = 16

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

from scipy import interpolate
import numpy as np
from KS import *
from plotting import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 22
nu   = 1
dt   = 0.25
tTransient = 50
tEnd       = 550
tSim       = tEnd-tTransient
dns = KS(L=L, N=N1, dt=dt, nu=nu, tend=tTransient)

## simulate transient period
dns.simulate()
# convert to physical space
dns.fou2real()

u_restart = dns.uu[-1,:].copy()
v_restart = dns.vv[-1,:].copy()

## simulate rest
dns.IC( v0 = v_restart )
dns.simulate( nsteps=int(tSim/dt), restart=True )

# convert to physical space
dns.fou2real()
dns.compute_Ek()
dns.compute_Sgs(N2)

# get solution
u1 = dns.uu.copy()

## for plotting
uTruth = dns.uu
tTruth = dns.tt
xTruth = dns.x
sTruth, nTruth = np.meshgrid(np.arange(uTruth.shape[0])*dt, 2*np.pi*L/N1*(np.array(range(N1))+1))

#------------------------------------------------------------------------------
## restart
f_restart = interpolate.interp1d(xTruth, u_restart)

# restart from coarse physical space
sgs = KS(L=L, N=N2, dt=dt, nu=nu, tend=tSim)
sgs.setGroundTruth(tTruth, xTruth, uTruth)

# create interpolated IC
xCoarse = sgs.x
uRestartCoarse = f_restart(xCoarse)
vRestartCoarse = np.concatenate((v_restart[:((N2+1)//2)], v_restart[-(N2-1)//2:])) * N2 / N1

#sgs.IC( u0 = uRestartCoarse )
sgs.IC( v0 = vRestartCoarse )

# continue simulation
sgs.simulate()

# convert to physical space
sgs.fou2real()
sgs.compute_Ek()

# get solution
uCoarse = sgs.uu

# eval truth on coarse Grid
uTruthToCoarse = sgs.mapGroundTruth()

#------------------------------------------------------------------------------
## plot comparison
fig, axs = plt.subplots(1,3)
s, n = np.meshgrid(sgs.tt, sgs.x)

cs0 = axs[0].contourf(sTruth, nTruth, uTruth.T, 50, cmap=plt.get_cmap("seismic"))
cs1 = axs[1].contourf(s, n, uCoarse.T, 50, cmap=plt.get_cmap("seismic"))
diff = np.abs(uCoarse-uTruthToCoarse)
cs2 = axs[2].contourf(s, n, diff.T, 50, cmap=plt.get_cmap("seismic"))

# plt.colorbar(cs0, ax=axs[0])
plt.colorbar(cs1, ax=axs[1])
plt.colorbar(cs2, ax=axs[2])
plt.setp(axs[:], xlabel='$t$')
plt.setp(axs[0], ylabel='$x$')
# for c in cs.collections: c.set_rasterized(True)
axs[1].set_yticklabels([])
axs[2].set_yticklabels([])
fig.savefig('interpolate.png')

makePlot(dns, sgs, sgs, "evolution", spectralReward=True)
