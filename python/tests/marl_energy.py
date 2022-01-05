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
tTransient = 10
tEnd = 50
dns = KS(L=L, N=N, dt=dt, nu=1.0, tend=tTransient)
xTruth = dns.x

## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

#------------------------------------------------------------------------------
# restart
u_restart = dns.uu[-1,:].copy()
f_restart = interpolate.interp1d(xTruth, u_restart)

dns.IC( u0 = u_restart )

# finsh dns
dns.simulate( nsteps=int( (tEnd-tTransient)/dt), restart=True )

# convert to physical space
dns.fou2real()

# compute energies
dns.compute_Ek()

# get dns energy
dnsEnergy = dns.Ek_t
print(dnsEnergy.size)

# restart from coarse physical space
N = 512
subgrid = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd-tTransient)

# create interpolated IC
xCoarse = subgrid.x
uRestartCoarse = f_restart(xCoarse)
subgrid.IC( u0 = uRestartCoarse )

print(uRestartCoarse)

# MARL LOOP
rewardHist = np.zeros((int((tEnd-tTransient)/dt)))
actions = 1+0.0001*np.random.normal(0, 1, size=(int((tEnd-tTransient)/dt), N))
for i in range(int((tEnd-tTransient)/dt)):
    subgrid.updateField(actions[i,:])
    subgrid.step()
    subgrid.fou2real()

    # compute energy at step i 
    subgridEnergy = 0.5*np.sum(subgrid.uu[i,:]**2)*subgrid.dx

    # calculate reward
    rewardHist[i] = -abs(subgridEnergy - dnsEnergy[i])

# compute energy from u field
subGridEnergyAll = 0.5*np.sum(subgrid.uu**2,axis=1)*subgrid.dx

# double check reward calculation
rewardAll = -abs(subGridEnergyAll-dnsEnergy)

#------------------------------------------------------------------------------
## plot comparison
fig, axs = plt.subplots(1,1)

axs.plot(rewardHist)
axs.plot(rewardAll)
fig.savefig('marl_energies.png')
