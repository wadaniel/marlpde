#!/bin/python3

"""
This scripts simulates the Diffusion equation on a grid (N) until t=tEnd. The 
initial condition is set to be approx k^-5/3.
"""

import numpy as np

# Discretization grid
N = 2048
NRUNS = 64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

from plotting import *
from Burger import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L       = 2*np.pi
dt      = 0.0001
tEnd    = 5
nu      = 0.02
ic      = 'turbulence'
noise   = 3.
seed    = 42
forcing = True

ntimestep = int(tEnd / dt)
spectra = np.zeros((NRUNS, ntimestep+1, N))

#------------------------------------------------------------------------------
## simulate
for i in range(NRUNS):
    print("Simulate ({}/{}) ..".format(i,NRUNS))
    
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, forcing=forcing, noise=noise, seed=seed)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()

    spectra[i, :, :] = dns.Ek_ktt


meanSpec = np.mean(spectra, axis=0)
sdevSpec = np.std(spectra, axis=0)

k = np.arange(N//2)
lb = meanSpec[-1, k] - sdevSpec[-1, k]
ub = meanSpec[-1, k] + sdevSpec[-1, k]

print(meanSpec.shape)
print(meanSpec)
print(sdevSpec)

fname = "spec_{}.npz".format(N)
np.savez(file=fname, meanSpec=meanSpec, sdevSpec=sdevSpec, N=N, dt=dt, nu=nu)

#------------------------------------------------------------------------------
## plot

figname = "mean_spectrum_{}.png".format(N)
fig, ax = plt.subplots(1,1)
ax.plot(k, meanSpec[-1,k])
ax.fill_between(k, lb, ub, alpha=0.25)
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig(figname)
