#!/bin/python3

"""
This scripts simulates the KS on a fine grid (N) up to t=tEnd. We plot the KS and
the energy spectra.
"""

# Discretization grid
N = 512

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
dt   = 0.05
tEnd = 3000
dns  = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd)

#------------------------------------------------------------------------------
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

#------------------------------------------------------------------------------
## plot result
u = dns.uu

fig, axs = plt.subplots(1,4)

s, n = np.meshgrid(2*np.pi*L/N*(np.array(range(N))+1), np.arange(tEnd/dt+1)*dt)

cs0 = axs[0].contourf(s, n, u, 50) # cmap=plt.get_cmap("seismic"))

fig.savefig('simulate.png')
