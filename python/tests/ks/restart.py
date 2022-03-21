#!/bin/python3

"""
This scripts simulates the KS on a grid (N) until t=tTransient. The solution 
from the ast time-step is extracted and then taken as the initial condidtion for
two more runs (i) in real space and ind (ii) fourier space with simulation length
tEnd (until t=tEnd+tTransient). The transient phase and both results and the difference is plotted.
"""

# Discretization grid
N = 512

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from KS import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 22
dt   = 0.05
tTransient = 50
tEnd       = 3000
tSim       = tEnd-tTransient
dns = KS(L=L, N=N, dt=dt, nu=1.0, tend=tTransient)

#------------------------------------------------------------------------------
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

#------------------------------------------------------------------------------
## plot result
u = dns.uu
#------------------------------------------------------------------------------
## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()

#############################
# restart from physical space
dns.IC( u0 = u_restart )

# continue simulation
dns.simulate( nsteps=int(tSim/dt), restart=True )

# convert to physical space
dns.fou2real()

# get solution
u1 = dns.uu

#############################
# restart from Fourier space
dns.IC( v0 = v_restart )

# continue simulation
dns.simulate( nsteps=int(tSim/dt), restart=True )

# convert to physical space
dns.fou2real()

# get solution
u2 = dns.uu

#------------------------------------------------------------------------------
## plot comparison
print("Plotting restart.png ...")
fig, axs = plt.subplots(1,3)

cs0 = axs[0].contourf(dns.tt, dns.x, u1.T, 50, cmap=plt.get_cmap("seismic"))
cs1 = axs[1].contourf(dns.tt, dns.x, u2.T, 50, cmap=plt.get_cmap("seismic"))
diff = np.abs(u1-u2)
cs2 = axs[2].contourf(dns.tt, dns.x, diff.T, 50, cmap=plt.get_cmap("seismic"))

plt.colorbar(cs1, ax=axs[0])
plt.colorbar(cs2, ax=axs[2])
plt.setp(axs[:], xlabel='$t$')
plt.setp(axs[0], ylabel='$x$')
axs[1].set_yticklabels([])
axs[2].set_yticklabels([])
# for c in cs.collections: c.set_rasterized(True)
fig.savefig('restart.png')
plt.close()

print("Plotting kursiv.png ...")
fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
for i in range(16):
    t = int(i * tSim / dt / 16)
    k = int(i / 4)
    l = i % 4
    axs[k,l].plot(dns.x, u1[t,:])

fig.savefig('kursiv.png'.format())
plt.close()
