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
dt   = 0.25
tTransient = 10
tEnd       = 40 + tTransient  #50000
dns = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd)

## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

## plot result
uTruth = dns.uu
s, n = np.meshgrid(np.arange(uTruth.shape[0])*dt, 2*np.pi*L/N*(np.array(range(N))+1))

#------------------------------------------------------------------------------
## restart
x_truth = dns.x
u_restart = dns.uu[0,:].copy()
f_restart = interpolate.interp1d(x_truth, u_restart)

# restart from coarse physical space
N = 512
subgrid = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd)

subgrid.setGroundTruth(uTruth)

#
x_coarse = subgrid.x
u_restart_coarse = f_restart(x_coarse)
subgrid.IC( u0 = u_restart_coarse)

# continue simulation
subgrid.simulate( nsteps=int(tEnd/dt), restart=True )

# convert to physical space
subgrid.fou2real()

# get solution
uCoarse = subgrid.uu

# eval truth on coarse Grid
uFine = subgrid.mapGroundTruth()

#------------------------------------------------------------------------------
## plot comparison
fig, axs = plt.subplots(1,3)
s, n = np.meshgrid(np.arange(tEnd/dt+1)*dt, 2*np.pi*L/N*(np.array(range(N))+1))

cs0 = axs[0].contourf(s, n, uTruth.T, 50, cmap=plt.get_cmap("seismic"))
cs1 = axs[1].contourf(s, n, uCoarse.T, 50, cmap=plt.get_cmap("seismic"))
#diff = np.abs(uCoarse-uFine)
diff = np.abs(uTruth-uFine)
cs2 = axs[2].contourf(s, n, diff.T, 50, cmap=plt.get_cmap("seismic"))

# plt.colorbar(cs0, ax=axs[0])
plt.colorbar(cs1, ax=axs[1])
plt.colorbar(cs2, ax=axs[2])
plt.setp(axs[:], xlabel='$t$')
plt.setp(axs[0], ylabel='$x$')
# for c in cs.collections: c.set_rasterized(True)
fig.savefig('interpolate.png')
