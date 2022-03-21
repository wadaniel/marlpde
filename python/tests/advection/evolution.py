#!/bin/python3

"""
This scripts simulates the Diffusion equation on a grid (N) until t=tEnd. The 
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
from Advection import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 2*np.pi
dt   = 0.0001
tEnd = 10
nu   = 1.0
dns = Advection(L=L, N=N, dt=dt, nu=nu, case='box', tend=tEnd)

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

fig.savefig('evolution.png')
plt.close()

print("Plotting advection_evolution.png ...")
fig, axs = plt.subplots(4,4, sharex=True, sharey=False, figsize=(15,15))
for i in range(16):
    t = i * tEnd / 16
    tidx = int(t/dt)
    k = int(i / 4)
    l = i % 4
    axs[k,l].plot(dns.x, dns.uu[tidx,:])
    axs[k,l].plot(dns.x, dns.getAnalyticalSolution(t), '--k')

fig.savefig('advection_evolution.png'.format())
plt.close()
