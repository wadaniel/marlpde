#!/bin/python3

"""
This scripts simulates the KS on a fine grid (N) up to t=tEnd. We plot the KS and
the instanteneous energy plus the time averaged energy, and the energy spectra at 
start, mid and end of the simulation.
"""

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
# DNS defaults
N    = 1024
L    = 22/(2*np.pi)
nu   = 1.0
dt   = 0.01
tTransient = 0
tEnd = 5000
tSim = tEnd - tTransient
nSimSteps = int(tSim/dt)

#------------------------------------------------------------------------------
## transient
dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient)
dns.simulate()
dns.fou2real()
 
#------------------------------------------------------------------------------
## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()
 
# set IC
dns.IC( v0 = v_restart )

# continue simulation
dns.simulate( nsteps=int(tSim/dt), restart = True )

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

fig, axs = plt.subplots(1,3, figsize=(15,15))
s, n = np.meshgrid(2*np.pi*L/N*(np.array(range(N))+1), dns.tt)

axs[0].contourf(s, n, u, 50)
axs[1].plot(dns.tt, e_t)
axs[1].plot(dns.tt, e_tt)
axs[2].plot(k, 2.0/N * np.abs(e_ktt[0,0:N//2]),'b:')
axs[2].plot(k, 2.0/N * np.abs(e_ktt[nSimSteps//2,0:N//2]),'b--')
axs[2].plot(k, 2.0/N * np.abs(e_ktt[-1,0:N//2]),'b')
axs[2].set_xscale('log')
axs[2].set_yscale('log')

fig.savefig('simulate.png')
