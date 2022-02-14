#!/bin/python3

"""
This scripts simulates the Burger equation on a grid (N) until t=tTransient. The solution 
from the ast time-step is extracted and then taken as the initial condidtion for
two more runs (i) in real space and ind (ii) fourier space with simulation length
tEnd. 
"""

# Discretization grid
N = 1024

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from Burger import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 2*np.pi
dt   = 0.0005
tEnd = 5
nu   = 0.01
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd)

#------------------------------------------------------------------------------
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

#------------------------------------------------------------------------------
## plot result
u = dns.uu

## restart
v_restart = dns.vv[0,:].copy()
u_restart = dns.uu[0,:].copy()

#------------------------------------------------------------------------------
print("Simulate..")
#############################
# restart from physical space
dns.IC( u0 = u_restart )

# continue simulation
dns.simulate( nsteps=int(tEnd/dt), restart=True )

# convert to physical space
dns.fou2real()

# get solution
u1 = dns.uu

#############################
# restart from Fourier space
dns.IC( v0 = v_restart )

# continue simulation
dns.simulate( nsteps=int(tEnd/dt), restart=True )

# convert to physical space
dns.fou2real()

# get solution
u2 = dns.uu

#------------------------------------------------------------------------------
## plot comparison
print("Plotting restart.png ...")
sre, nre = np.meshgrid( np.arange(tEnd/dt+1)*dt, 2*np.pi*L/N*(np.array(range(N))+1))
fig, axs = plt.subplots(1,3)

cs0 = axs[0].contourf(sre, nre, u1.T, 50, cmap=plt.get_cmap("seismic"))
cs1 = axs[1].contourf(sre, nre, u2.T, 50, cmap=plt.get_cmap("seismic"))
diff = np.abs(u1-u2)
cs2 = axs[2].contourf(sre, nre, diff.T, 50, cmap=plt.get_cmap("seismic"))

plt.colorbar(cs1, ax=axs[0])
plt.colorbar(cs2, ax=axs[2])
plt.setp(axs[:], xlabel='$t$')
plt.setp(axs[0], ylabel='$x$')
axs[1].set_yticklabels([])
axs[2].set_yticklabels([])
# for c in cs.collections: c.set_rasterized(True)
fig.savefig('restart.png')
plt.close()

print("Plotting burger.png ...")
fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
for i in range(16):
    t = int(i * tEnd / dt / 16)
    k = int(i / 4)
    l = i % 4
    axs[k,l].plot(nre, u1[t,:])

fig.savefig('burger.png'.format())
plt.close()
