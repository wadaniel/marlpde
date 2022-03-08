#!/bin/python3

"""
This scripts simulates the Burger equation on a grid (N) until t=tEnd. The 
initial condition is set to be approx k^-5/3.
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
dt   = 0.001
tEnd = 5
nu   = 0.01
ic   = 'turbulence'


dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic)

print(v0)
print(v0[:10])

# convert to physical space
dns.IC(v0 = v0)

print("Plot IC..")
fig, axs = plt.subplots(1,2) #figsize=(15,15))
axs[0].plot(dns.u0)
axs[1].plot(dns.v0)
axs[1].set_xscale('log')
axs[1].set_yscale('log')

#------------------------------------------------------------------------------
print("Simulate..")
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

#------------------------------------------------------------------------------
## plot

print("Plotting turbulence.png ...")
sre, nre = np.meshgrid( np.arange(tEnd/dt+1)*dt, 2*np.pi*L/N*(np.array(range(N))+1))
fig, axs = plt.subplots(1,1)

cs0 = axs.contourf(sre, nre, dns.uu.T, 50, cmap=plt.get_cmap("seismic"))

plt.colorbar(cs0, ax=axs)

# for c in cs.collections: c.set_rasterized(True)
fig.savefig('turbulence.png')
plt.close()

print("Plotting burger_turbulence.png ...")
fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
for i in range(16):
    t = int(i * tEnd / dt / 16)
    k = int(i / 4)
    l = i % 4
    axs[k,l].plot(nre, dns.uu[t,:])

fig.savefig('burger_turbulence.png'.format())
plt.close()
