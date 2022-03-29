#!/bin/python3

"""
This scripts simulates the Burger on two grids (N1, N2) up to t=tEnd. We plot the Burger and
the instanteneous energy plus the time averaged energy, and the energy spectra at 
start, mid and end of the simulation.
"""

import math

# Discretization grid
N1 = 1024
N2 = 1024

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
import time
from scipy import interpolate
from scipy.fft import fftfreq

from Burger import *
from Burger_rk import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 2*np.pi
#dt   = 0.001
dt   = 0.0001
tEnd = 2.5
nu   = 0.01
nt   = int(tEnd/dt)
ic   = 'sinus'
seed = 31

dns = Burger(L=L, N=N1, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed, noise=0.)

#------------------------------------------------------------------------------
print("Simulate with spectral solver")
start = time.time()
## simulate DNS in transient phase and produce IC
dns.simulate()
end = time.time()
print("Done in {}s".format(end-start))
# convert to physical space
dns.fou2real()
# compute energies
dns.compute_Ek()
# IC and interpolation
IC = dns.u0.copy()
f_IC = interpolate.interp1d(dns.x, IC, kind='cubic')

#------------------------------------------------------------------------------
print("Simulate with RK solver")
## simulate SGS from IC
dns_rk = Burger_rk(L=L, N=N1, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed, noise=0.)
#dns_rk = Burger_rk(L=L, N=N2, dt=dt, nu=nu, tend=tEnd)
#u0 = f_IC(dns_rk.x)
#dns_rk.IC(u0 = u0)
#dns_rk.IC(v0 = dns.v0[:N2]*N2/N1)
dns_rk.setGroundTruth(dns.tt, dns.x, dns.uu)

start = time.time()
dns_rk.simulate()
end = time.time()
print("Done in {}s".format(end-start))

# convert to fourier space
dns_rk.real2fou()
# compute energies
dns_rk.compute_Ek()

#------------------------------------------------------------------------------
## compute errors

# instantaneous energy error
errEk_t = np.abs(dns.Ek_t - dns_rk.Ek_t)
# time-cumulative energy average error as a function of time
errEk_tt = np.abs(dns.Ek_tt - dns_rk.Ek_tt)
# Time-averaged cum energy spectrum as a function of wavenumber
errEk_ktt = ((dns.Ek_ktt[:, :N2] - dns_rk.Ek_ktt[:, :N2])**2).mean(axis=1)

# eval truth on coarse grid
uTruthCoarse = dns_rk.mapGroundTruth()
errU_t = ((dns_rk.uu - uTruthCoarse)**2).mean(axis=1)

#------------------------------------------------------------------------------
## plot result
figName0 = 'simulate_energies_rk.png'
print("Plotting {} ...".format(figName0))

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

k2 = dns_rk.k[:N2//2]

axs[1,0].contourf(dns_rk.x, dns_rk.tt, dns_rk.uu, c0.levels)

axs[1,1].plot(dns_rk.tt, dns_rk.Ek_t)
axs[1,1].plot(dns_rk.tt, dns_rk.Ek_tt)

axs[1,2].plot(k2, np.abs(dns_rk.Ek_ktt[0,0:N2//2]),'b:')
axs[1,2].plot(k2, np.abs(dns_rk.Ek_ktt[nt//2,0:N2//2]),'b--')
axs[1,2].plot(k2, np.abs(dns_rk.Ek_ktt[-1,0:N2//2]),'b')

axs[1,2].plot(k2, np.abs(dns.Ek_ktt[0,0:N2//2] - dns_rk.Ek_ktt[0,0:N2//2]),'r:')
axs[1,2].plot(k2, np.abs(dns.Ek_ktt[nt//2,0:N2//2] - dns_rk.Ek_ktt[nt//2,0:N2//2]),'r--')
axs[1,2].plot(k2, np.abs(dns.Ek_ktt[-1,0:N2//2] - dns_rk.Ek_ktt[-1,0:N2//2]),'r')

axs[1,2].set_xscale('log')
axs[1,2].set_yscale('log')
fig.savefig(figName0)

#------------------------------------------------------------------------------
figName1 = 'simulate_ediff_rk.pdf'
print("Plotting {} ...".format(figName1))

fig, axs = plt.subplots(1,4, sharex='row', figsize=(15,15)) #, sharey='row')

axs[0].title.set_text('Instant energy err')
axs[0].plot(dns_rk.tt, errEk_t)
axs[0].set_yscale('log')
axs[1].title.set_text('Time avg energy err')
axs[1].plot(dns_rk.tt, errEk_tt)
axs[1].set_yscale('log')
axs[2].title.set_text('Time avg energy spec mse err')
axs[2].plot(dns_rk.tt, errEk_ktt)
axs[2].set_yscale('log')
axs[3].title.set_text('Instant field mse err')
axs[3].plot(dns_rk.tt, errU_t)
axs[3].set_yscale('log')

fig.savefig(figName1)

#------------------------------------------------------------------------------
figName2 = 'simulate_evolution_rk.pdf'
print("Plotting {} ...".format(figName2))

fig2, axs = plt.subplots(4,4, sharex=True, sharey=False, figsize=(15,15))
for i in range(16):
    t = i * tEnd / 16
    tidx = int(t/dt)
    k = int(i / 4)
    l = i % 4
    
    axs[k,l].plot(dns.x, dns.uu[tidx,:], '--k')
    axs[k,l].plot(dns_rk.x, dns_rk.uu[tidx,:], 'steelblue')

fig2.savefig(figName2)
