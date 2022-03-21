#!/bin/python3

"""
This scripts simulates the Diffusion equation on a grid (N) until t=tEnd. The 
initial condition is set to be approx k^-5/3.
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
from Burger import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.01
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case='turbulence', noisy=True)

#------------------------------------------------------------------------------
print("Simulate..")
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

#------------------------------------------------------------------------------
## plot

print("Plotting evolution.png ...")
fig, axs = plt.subplots(1,1)

cs0 = axs.contourf(dns.tt, dns.x, dns.uu.T, 50, cmap=plt.get_cmap("seismic"))

plt.colorbar(cs0, ax=axs)

# for c in cs.collections: c.set_rasterized(True)
fig.savefig('evolution.png')
plt.close()

print("Plotting burger_evolution.png ...")
fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
for i in range(16):
    t = int(i * tEnd / dt / 16)
    k = int(i / 4)
    l = i % 4
    axs[k,l].plot(dns.x, dns.uu[t,:])

fig.savefig('burger_evolution.png'.format())
plt.close()
