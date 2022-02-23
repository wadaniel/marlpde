#!/bin/python3

"""
This scripts simulates the Burger on two grids (N1, N2) up to t=tEnd. We plot the Burger and
the instanteneous energy plus the time averaged energy, and the energy spectra at 
start, mid and end of the simulation.
"""

import math

# Discretization grid
N1 = 1024
N2 = 64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from scipy import interpolate
from scipy.fft import fftfreq

from Burger import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 2*np.pi
dt   = 0.0005
tEnd = 5
nu   = 0.01

dns = Burger(L=L, N=N1, dt=dt, nu=nu, tend=tEnd)

#------------------------------------------------------------------------------
print("Simulate DNS")
## simulate DNS in transient phase and produce IC
dns.simulate()
# convert to physical space
dns.fou2real()
# compute energies
dns.compute_Ek()
# IC and interpolation
IC = dns.u0.copy()
f_IC = interpolate.interp1d(dns.x, IC)

#------------------------------------------------------------------------------
print("Simulate SGS")
## simulate SGS from IC
sgs = Burger(L=L, N=N2, dt=dt, nu=nu, tend=tEnd)
u0 = f_IC(sgs.x)
sgs.IC(u0 = u0)

sgs.simulate()
# convert to physical space
sgs.fou2real()
# compute energies
sgs.compute_Ek()

#------------------------------------------------------------------------------
## compute errors

# instantaneous energy error
errEk_t = dns.Ek_t - sgs.Ek_t
# time-cumulative energy average error as a function of time
errEk_tt = dns.Ek_tt - sgs.Ek_tt
# Time-averaged energy spectrum as a function of wavenumber
errEk_ktt = ((dns.Ek_ktt[:, :N2] - sgs.Ek_ktt[:, :N2])**2).mean(axis=1)

#------------------------------------------------------------------------------
## plot result
print("plot energy")
k1 = dns.k[:N1//2]

time = np.arange(tEnd/dt+1)*dt
s, n = np.meshgrid(2*np.pi*L/N1*(np.array(range(N1))+1), time)

fig, axs = plt.subplots(2,3, sharex='col', sharey='col')
axs[0,0].contourf(s, n, dns.uu, 50)

axs[0,1].plot(time, dns.Ek_t)
axs[0,1].plot(time, dns.Ek_tt)

axs[0,2].plot(k1, np.abs(dns.Ek_ktt[0,0:N1//2]),'b--')
axs[0,2].plot(k1, np.abs(dns.Ek_ktt[tEnd//2,0:N1//2]),'b:')
axs[0,2].plot(k1, np.abs(dns.Ek_ktt[-1,0:N1//2]),'b')
axs[0,2].set_xscale('log')
axs[0,2].set_yscale('log')

k2 = sgs.k[:N2//2]
s, n = np.meshgrid(2*np.pi*L/N2*(np.array(range(N2))+1), time)

axs[1,0].contourf(s, n, sgs.uu, 50)

axs[1,1].plot(time, sgs.Ek_t)
axs[1,1].plot(time, sgs.Ek_tt)

axs[1,2].plot(k2, np.abs(dns.Ek_ktt[0,0:N2//2]),'b--')
axs[1,2].plot(k2, np.abs(dns.Ek_ktt[tEnd//2,0:N2//2]),'b:')
axs[1,2].plot(k2, np.abs(dns.Ek_ktt[-1,0:N2//2]),'b')
axs[1,2].set_xscale('log')
axs[1,2].set_yscale('log')

fig.savefig('simulate_energies.png')

fig, axs = plt.subplots(1,3, sharex='col', sharey='col')

axs[0].plot(time, errEk_t)
axs[1].plot(time, errEk_tt)
axs[2].plot(time, errEk_ktt)

fig.savefig('simulate_ediff.png')
