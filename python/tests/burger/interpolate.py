#!/bin/python3

"""
This scripts simulates the Burgers on a fine grid (N1) and on a coare grid (N2)
The IC in the coarse grid is the interpolated IC of the fine grid. Then we plot 
the Burgers of the fine grid, the Burgers on the coarse grid, and the difference between 
Burgers of fine grid interpolated on coarse grid vs Burgers on coarse grid.

"""

# Discretization fine grid (DNS)
N1 = 1024

# Discretization coarse grid
N2 = 512

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

from scipy import interpolate
import numpy as np
from Burger import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 2*np.pi
dt   = 0.0001
tEnd = 5
nu   = 0.01
ic   = 'turbulence'
seed = 31

dns = Burger(L=L, N=N1, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed)

print("simulate dns..")
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

## for plotting
uTruth = dns.uu
tTruth = dns.tt
xTruth = dns.x

#------------------------------------------------------------------------------
## restart
u_restart = dns.uu[0,:].copy()
f_restart = interpolate.interp1d(xTruth, u_restart)

# restart from coarse physical space
subgrid = Burger(L=L, N=N2, dt=dt, nu=nu, tend=tEnd)
subgrid.IC(case=ic)
subgrid.setGroundTruth(tTruth, xTruth, uTruth)

# create interpolated IC
xCoarse = subgrid.x
uRestartCoarse = f_restart(xCoarse)
subgrid.IC( u0 = uRestartCoarse)

# continue simulation
print("simulate sgs..")
subgrid.simulate( nsteps=int(tEnd/dt), restart=True )

# convert to physical space
subgrid.fou2real()

# get solution
uCoarse = subgrid.uu

# eval truth on coarse grid
uTruthToCoarse = subgrid.mapGroundTruth()

#------------------------------------------------------------------------------
## plot comparison
print("plotting..")

umax = max(dns.uu.max(), subgrid.uu.max())
umin = min(dns.uu.min(), subgrid.uu.min())
ulevels = np.linspace(umin, umax, 50)

fig, axs = plt.subplots(1,3, figsize=(15,15))
cs0 = axs[0].contourf(dns.x, dns.tt, uTruth, ulevels, cmap=plt.get_cmap("coolwarm"))
cs1 = axs[1].contourf(subgrid.x, subgrid.tt, uCoarse, ulevels, cmap=plt.get_cmap("coolwarm"))
diff = np.abs(uCoarse-uTruthToCoarse)
cs2 = axs[2].contourf(subgrid.x, subgrid.tt, diff, 50, cmap=plt.get_cmap("coolwarm"))

plt.setp(axs[:], xlabel='$x$')
plt.setp(axs[0], ylabel='$t$')
# for c in cs.collections: c.set_rasterized(True)
axs[1].set_yticklabels([])
axs[2].set_yticklabels([])
fig.savefig('interpolate.png')
