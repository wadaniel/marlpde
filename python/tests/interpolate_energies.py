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
N    = 4096
dt   = 0.25
tEnd = 100
dns = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd)

## simulate DNS
dns.simulate()
# convert to physical space
dns.fou2real()
# compute energy spectrum
dns.compute_Ek()

# extract domain and solution
uTruth = dns.uu
eTruth = dns.Ek_t
eTruthU = 0.5*np.sum(dns.uu**2, axis=1)*dns.dx
tTruth = dns.tt
xTruth = dns.x
sTruth, nTruth = np.meshgrid(np.arange(uTruth.shape[0])*dt, 2*np.pi*L/N*(np.array(range(N))+1))

# create interpolated truth function
f_interpolate = interpolate.interp2d(xTruth, tTruth, dns.uu)

#------------------------------------------------------------------------------

## restart
u_restart = dns.uu[0,:].copy()
f_restart = interpolate.interp1d(xTruth, u_restart)

# restart from coarse physical space
N = 32
subgrid = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd)
subgrid.setGroundTruth(tTruth, xTruth, uTruth)

# create interpolated IC
xCoarse = subgrid.x
uRestartCoarse = f_restart(xCoarse)

# set interpolated
subgrid.IC( u0 = uRestartCoarse)

# continue simulation
subgrid.simulate( nsteps=int(tEnd/dt), restart=True )

# convert to physical space
subgrid.fou2real()

# get solution
uCoarse = subgrid.uu

# eval truth on coarse Grid
uTruthToCoarse = subgrid.mapGroundTruth()

# compute energy spectrum
subgrid.compute_Ek()

eCoarse = subgrid.Ek_t
eCoarseU = 0.5*np.sum(uCoarse**2,axis=1)*subgrid.dx

# compute energy spectrum of interpoalted truth
uTruthInterp = f_interpolate(xCoarse, tTruth)
eTruthInterp = 0.5*np.sum(uTruthInterp**2,axis=1)*subgrid.dx

# compute differences
diffTruth = eTruth-eCoarse
diffInterpTruth = eTruthInterp-eCoarse
diffTruths = eTruth-eTruthInterp

#------------------------------------------------------------------------------

## plot comparison
fig, axs = plt.subplots(1,3)

axs[0].plot(eTruth)
axs[0].plot(eTruthU)
axs[0].plot(eTruthInterp)
axs[1].plot(eCoarse)
axs[1].plot(eCoarseU)
axs[2].plot(diffTruths)
axs[2].plot(diffTruth)
axs[2].plot(diffInterpTruth)

# plt.colorbar(cs0, ax=axs[0])
plt.setp(axs[:], xlabel='$t$')
plt.setp(axs[0], ylabel='$e$')
# for c in cs.collections: c.set_rasterized(True)
axs[1].set_yticklabels([])
fig.savefig('interpolate_energies.png')
