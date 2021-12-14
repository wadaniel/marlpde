#!/bin/python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../_model/')

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

#------------------------------------------------------------------------------
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

#------------------------------------------------------------------------------
## plot result
u = dns.uu
s, n = np.meshgrid(np.arange(u.shape[0])*dt, 2*np.pi*L/N*(np.array(range(N))+1))
print(u.shape)
#print(s.shape)
print(n)
cs = plt.contourf(s, n, u.T, 50, cmap=plt.get_cmap("seismic"), levels=np.linspace(-10,10,21))
plt.colorbar()
plt.ylabel(r"$x$")
plt.xlabel(r"$t$")
#for c in cs.collections: c.set_rasterized(True)
# plt.show()

#------------------------------------------------------------------------------
## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()

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
fig, axs = plt.subplots(1,4)
s, n = np.meshgrid(np.arange(tEnd/dt+1)*dt, 2*np.pi*L/N*(np.array(range(N))+1))

cs0 = axs[0].contourf(s, n, u.T, 50, cmap=plt.get_cmap("seismic"))
cs0 = axs[1].contourf(s, n, u1.T, 50, cmap=plt.get_cmap("seismic"))
cs1 = axs[2].contourf(s, n, u2.T, 50, cmap=plt.get_cmap("seismic"))
diff = np.abs(u1-u2)
cs2 = axs[3].contourf(s, n, diff.T, 50, cmap=plt.get_cmap("seismic"))

# plt.colorbar(cs0, ax=axs[0])
plt.colorbar(cs1, ax=axs[1])
plt.colorbar(cs2, ax=axs[2])
plt.setp(axs[:], xlabel='$t$')
plt.setp(axs[0], ylabel='$x$')
# for c in cs.collections: c.set_rasterized(True)
fig.savefig('restart.png')
