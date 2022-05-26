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

from Kuramoto import *

#------------------------------------------------------------------------------
# DNS defaults
N    = 32
L    = 2*np.pi
nu   = 100
dt   = 3*1e-6
tTransient = 6*1e-3
tEnd = 6*1e-3 + tTransient
tSim = tEnd - tTransient
# dt   = 0.01
# tSim = 100
nSimSteps = int(tSim/dt)
Cs = 0.001

#------------------------------------------------------------------------------
## transient
print("simulate transient")

dns = Kuramoto(L=L, N=N, dt=dt, nu=nu, tend=tTransient, case='ETDRK4')
dns.simulate()
dns.fou2real()
dns.compute_Ek()

#------------------------------------------------------------------------------
## restart
# v_restart = dns.vv[-1,:].copy()
# u_restart = dns.uu[-1,:].copy()
#
# print("simulate DNS")
# # set IC
# dns.IC( v0 = v_restart )
# # continue simulation
# dns.simulate( nsteps=int(tSim/dt), restart = True)
# # convert to physical space
# dns.fou2real()
# # compute energies
# dns.compute_Ek()

#------------------------------------------------------------------------------

# plot result
u = dns.uu
e_t = dns.Ek_t
e_tt = dns.Ek_tt
e_ktt = dns.Ek_ktt

k = dns.k[:N//2]

#fig, axs = plt.subplots(1,3, figsize=(15,15))
fig, axs = plt.subplots(1,3)
print(dns.tt.shape)
print(dns.x.shape)
print(dns.uu.shape)

axs[0].contourf(dns.x, dns.tt, dns.uu)
axs[1].plot(dns.tt, e_t)
axs[1].plot(dns.tt, e_tt)
axs[2].plot(k, 2.0/N * np.abs(e_ktt[0,0:N//2]),'b:')
axs[2].plot(k, 2.0/N * np.abs(e_ktt[nSimSteps//2,0:N//2]),'b--')
axs[2].plot(k, 2.0/N * np.abs(e_ktt[-1,0:N//2]),'b')
axs[2].set_xscale('log')
axs[2].set_yscale('log')

print("Plotting simulate.png")
fig.savefig('simulate.png')
