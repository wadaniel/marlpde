#!/bin/python3

"""
This scripts simulates the KS on a fine grid (N) up to t=tEnd. We plot the KS and
the instanteneous energy plus the time averaged energy, and the energy spectra at 
start, mid and end of the simulation.
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
from scipy.fft import fftfreq

from KS import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 22/(2*np.pi)
dt   = 0.05
tEnd = 1000
dns  = KS(L=L, N=N, dt=dt, nu=1.0, tend=tEnd)

#------------------------------------------------------------------------------
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()
# compute energies
dns.compute_Ek()

#------------------------------------------------------------------------------
## plot result
u = dns.uu
e_t = dns.Ek_t
e_tt = dns.Ek_tt
e_ktt = dns.Ek_ktt

k = dns.k[:N//2]

time = np.arange(tEnd/dt+1)*dt

fig, axs = plt.subplots(1,3)
s, n = np.meshgrid(2*np.pi*L/N*(np.array(range(N))+1), time)

axs[0].contourf(s, n, u, 50)
axs[1].plot(time, e_t)
axs[1].plot(time, e_tt)
axs[2].plot(k, 2.0/N * np.abs(e_ktt[0,0:N//2]),'b--')
axs[2].plot(k, 2.0/N * np.abs(e_ktt[tEnd//2,0:N//2]),'b:')
axs[2].plot(k, 2.0/N * np.abs(e_ktt[-1,0:N//2]),'b')
axs[2].set_xscale('log')
axs[2].set_yscale('log')

fig.savefig('simulate.png')
