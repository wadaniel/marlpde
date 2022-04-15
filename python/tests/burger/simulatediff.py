#!/bin/python3

"""
This scripts simulates the Burger on two grids (N1, N2) up to t=tEnd. We plot the Burger and
the instanteneous energy plus the time averaged energy, and the energy spectra at 
start, mid and end of the simulation.
"""

import math

# Discretization grid
N1 = 512
N2 = 16

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
dt   = 0.001
tEnd = 5
nu   = 0.02
nt   = int(tEnd/dt)
ic   = 'turbulence'
seed = 42

dns = Burger(L=L, N=N1, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed, noise=0.)

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
f_IC = interpolate.interp1d(dns.x, IC, kind='cubic')

#------------------------------------------------------------------------------
print("Simulate SGS")
## simulate SGS from IC
sgs = Burger(L=L, N=N2, dt=dt, nu=nu, tend=tEnd)
#u0 = f_IC(sgs.x)
#sgs.IC(u0 = u0)

v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:]))
sgs.IC(v0 = v0*N2/N1)
sgs.setGroundTruth(dns.tt, dns.x, dns.uu)

sgs.simulate()
# convert to physical space
sgs.fou2real()
# compute energies
sgs.compute_Ek()

#------------------------------------------------------------------------------
## compute errors

# instantaneous energy error
errEk_t = np.abs(dns.Ek_t - sgs.Ek_t)
# time-cumulative energy average error as a function of time
errEk_tt = np.abs(dns.Ek_tt - sgs.Ek_tt)
# Time-averaged cum energy spectrum as a function of wavenumber
errEk_ktt = ((dns.Ek_ktt[:, :N2] - sgs.Ek_ktt[:, :N2])**2).mean(axis=1)

# eval truth on coarse grid
uTruthCoarse = sgs.mapGroundTruth()
errU_t = ((sgs.uu - uTruthCoarse)**2).mean(axis=1)

#------------------------------------------------------------------------------
## plot result
k1 = dns.k[:N1//2]

fig, axs = plt.subplots(2,3, sharex='col', sharey='col')
c0 = axs[0,0].contourf(dns.x, dns.tt, dns.uu, 50)

axs[0,1].plot(dns.tt, dns.Ek_t)
axs[0,1].plot(dns.tt, dns.Ek_tt)

axs[0,2].plot(k1, np.abs(dns.Ek_ktt[0,0:N1//2]),'b:')
axs[0,2].plot(k1, np.abs(dns.Ek_ktt[nt//2,0:N1//2]),'b--')
axs[0,2].plot(k1, np.abs(dns.Ek_ktt[-1,0:N1//2]),'b')
axs[0,2].plot(k1[2:-10], 1e-5*k1[2:-10]**(-2),'k--', linewidth=0.5)

axs[0,2].set_xscale('log')
axs[0,2].set_yscale('log')

k2 = sgs.k[:N2//2]

axs[1,0].contourf(sgs.x, sgs.tt, sgs.uu, c0.levels)

axs[1,1].plot(sgs.tt, sgs.Ek_t)
axs[1,1].plot(sgs.tt, sgs.Ek_tt)

axs[1,2].plot(k2, np.abs(sgs.Ek_ktt[0,0:N2//2]),'b:')
axs[1,2].plot(k2, np.abs(sgs.Ek_ktt[nt//2,0:N2//2]),'b--')
axs[1,2].plot(k2, np.abs(sgs.Ek_ktt[-1,0:N2//2]),'b')

axs[1,2].plot(k2, np.abs(dns.Ek_ktt[0,0:N2//2] - sgs.Ek_ktt[0,0:N2//2]),'r:')
axs[1,2].plot(k2, np.abs(dns.Ek_ktt[nt//2,0:N2//2] - sgs.Ek_ktt[nt//2,0:N2//2]),'r--')
axs[1,2].plot(k2, np.abs(dns.Ek_ktt[-1,0:N2//2] - sgs.Ek_ktt[-1,0:N2//2]),'r')

axs[1,2].set_xscale('log')
axs[1,2].set_yscale('log')

print("Plot simulate_energies.png")
fig.savefig('simulate_energies.png')

fig, axs = plt.subplots(1,4, sharex='row', figsize=(15,15)) #, sharey='row')

axs[0].title.set_text('Instant energy err')
axs[0].plot(sgs.tt, errEk_t)
axs[0].set_yscale('log')
axs[1].title.set_text('Time avg energy err')
axs[1].plot(sgs.tt, errEk_tt)
axs[1].set_yscale('log')
axs[2].title.set_text('Time avg energy spec mse err')
axs[2].plot(sgs.tt, errEk_ktt)
axs[2].set_yscale('log')
axs[3].title.set_text('Instant field mse err')
axs[3].plot(sgs.tt, errU_t)
axs[3].set_yscale('log')

print("plot simulate_ediff.png")
fig.savefig('simulate_ediff.png')

#------------------------------------------------------------------------------
figName2 = 'simulate_evolution.png'
print("Plotting {} ...".format(figName2))

fig2, axs = plt.subplots(4,4, sharex=True, sharey=False, figsize=(15,15))
for i in range(16):
    t = i * tEnd / 16
    tidx = int(t/dt)
    k = int(i / 4)
    l = i % 4
    
    axs[k,l].plot(dns.x, dns.uu[tidx,:], '--k')
    axs[k,l].plot(sgs.x, sgs.uu[tidx,:], 'steelblue')

fig2.savefig(figName2)
