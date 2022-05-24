#!/bin/python3

"""
This scripts simulates the Diffusion equation on a grid (N) until t=tEnd. The 
initial condition is set to be approx k^-5/3.
"""

# Discretization grid
N = 32

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from Diffusion import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.1
dns = Diffusion(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case='box')

#------------------------------------------------------------------------------
print("Simulate..")
## simulate
dns.simulate()

#------------------------------------------------------------------------------
## plot

print("Plotting evolution.png ...")
fig, axs = plt.subplots(1,1)

cs0 = axs.contourf(dns.tt, dns.x, dns.uu.T, 50, cmap=plt.get_cmap("seismic"))

plt.colorbar(cs0, ax=axs)

fig.savefig('evolution.png')
plt.close()

print("Plotting diffusion_evolution.png ...")
fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
for i in range(16):
    t = i * tEnd / 16
    tidx = int(t / dt)

    l = i % 4
    k = int(i / 4)
    sol = dns.getAnalyticalSolution(t)

    err = ((dns.uu[tidx,:] - sol)**2).mean()
    print(1e6*err)
    axs[k,l].plot(dns.x, dns.uu[tidx,:])
    axs[k,l].plot(dns.x, sol, 'k--')

fig.savefig('diffusion_evolution.png'.format())
plt.close()
