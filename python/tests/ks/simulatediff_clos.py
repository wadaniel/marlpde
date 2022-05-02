#!/bin/python3

"""
This scripts simulates the Burger on two grids (N, M) up to t=tEnd. We plot the Burger and
the instanteneous energy plus the time averaged energy, and the energy spectra at
start, mid and end of the simulation.
"""

import math

# Discretization grid
N = 1024
M = 32

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from scipy import interpolate
from scipy.fft import fftfreq

from KS import *
from KS_clos import *
#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 22
nu   = 1.0
dt   = 0.25
tTransient = 50
tEnd = 550
tSim = tEnd - tTransient
nt = int(tSim/dt)
basis = 'hat'
seed = 42
C = 0.1


dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient, seed=seed)

print("Simulate DNS")

dns.simulate()
dns.fou2real()
u_restart = dns.uu[-1,:].copy()
v_restart = dns.vv[-1,:].copy()

# simulate rest
dns.IC( u0 = u_restart)
dns.simulate( nsteps=int(tSim/dt), restart=True )
dns.fou2real()
dns.compute_Ek()

print(dns.ioutnum)
print(dns.tt.shape)
#IC and interpolation
IC = dns.u0.copy()
f_IC = interpolate.interp1d(dns.x, IC, kind='cubic')
#------------------------------------------------------------------------------
u_restart = dns.uu[0,:].copy()
v_restart = dns.vv[0,:].copy()

## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

# Initialize LES
les = KS_clos(L=L, N = M, dt=dt, nu=nu, tend=tSim, ssm=False, dsm=True, noise=0.)
v0 = np.concatenate((v_restart[:((M+1)//2)], v_restart[-(M-1)//2:])) * M / dns.N

les.IC( v0 = v0 )
les.setGroundTruth(dns.tt, dns.x, dns.uu)

print("Simulate LES")
les.simulate(C=C)
print(les.ioutnum)
print(les.tt.shape)
# convert to physical space
les.fou2real()
# compute energies
les.compute_Ek()

#------------------------------------------------------------------------------
## compute errors

# instantaneous energy error
errEk_t = np.abs(dns.Ek_t - les.Ek_t)
# time-cumulative energy average error as a function of time
errEk_tt = np.abs(dns.Ek_tt - les.Ek_tt)
# Time-averaged cum energy spectrum as a function of wavenumber
errEk_ktt = ((dns.Ek_ktt[:, :M] - les.Ek_ktt[:, :M])**2).mean(axis=1)

# eval truth on coarse grid
uTruthCoarse = les.mapGroundTruth()
errU_t = ((les.uu - uTruthCoarse)**2).mean(axis=1)

#------------------------------------------------------------------------------
## plot result
k1 = dns.k[:N//2]

fig, axs = plt.subplots(2,3, sharex='col', sharey='col')
c0 = axs[0,0].contourf(dns.x, dns.tt, dns.uu, 50)

axs[0,1].plot(dns.tt, dns.Ek_t)
axs[0,1].plot(dns.tt, dns.Ek_tt)

axs[0,2].plot(k1, np.abs(dns.Ek_ktt[0,0:N//2]),'b:')
axs[0,2].plot(k1, np.abs(dns.Ek_ktt[nt//2,0:N//2]),'b--')
axs[0,2].plot(k1, np.abs(dns.Ek_ktt[-1,0:N//2]),'b')
axs[0,2].plot(k1[2:-10], 1e-5*k1[2:-10]**(-2),'k--', linewidth=0.5)

axs[0,2].set_xscale('log')
axs[0,2].set_yscale('log')

k2 = les.k[:M//2]

axs[1,0].contourf(les.x, les.tt, les.uu, c0.levels)

axs[1,1].plot(les.tt, les.Ek_t)
axs[1,1].plot(les.tt, les.Ek_tt)

axs[1,2].plot(k2, np.abs(les.Ek_ktt[0,0:M//2]),'b:')
axs[1,2].plot(k2, np.abs(les.Ek_ktt[nt//2,0:M//2]),'b--')
axs[1,2].plot(k2, np.abs(les.Ek_ktt[-1,0:M//2]),'b')

axs[1,2].plot(k2, np.abs(dns.Ek_ktt[0,0:M//2] - les.Ek_ktt[0,0:M//2]),'r:')
axs[1,2].plot(k2, np.abs(dns.Ek_ktt[nt//2,0:M//2] - les.Ek_ktt[nt//2,0:M//2]),'r--')
axs[1,2].plot(k2, np.abs(dns.Ek_ktt[-1,0:M//2] - les.Ek_ktt[-1,0:M//2]),'r')

axs[1,2].set_xscale('log')
axs[1,2].set_yscale('log')

print("Plot simulate_energies.png")
fig.savefig('simulate_energies.png')

fig, axs = plt.subplots(1,4, sharex='row', figsize=(15,15)) #, sharey='row')

axs[0].title.set_text('Instant energy err')
axs[0].plot(les.tt, errEk_t)
axs[0].set_yscale('log')
axs[1].title.set_text('Time avg energy err')
axs[1].plot(les.tt, errEk_tt)
axs[1].set_yscale('log')
axs[2].title.set_text('Time avg energy spec mse err')
axs[2].plot(les.tt, errEk_ktt)
axs[2].set_yscale('log')
axs[3].title.set_text('Instant field mse err')
axs[3].plot(les.tt, errU_t)
axs[3].set_yscale('log')

print("plot simulate_ediff.png")
fig.savefig('simulate_ediff.png')

#------------------------------------------------------------------------------
figName2 = 'simulate_evolution.png'
print("Plotting {} ...".format(figName2))

fig2, axs = plt.subplots(4,4, sharex=True, sharey=False, figsize=(15,15))
for i in range(16):
    t = i * tSim / 16
    tidx = int(t/dt)
    k = int(i / 4)
    l = i % 4

    axs[k,l].plot(dns.x, dns.uu[tidx,:], '--k')
    axs[k,l].plot(les.x, les.uu[tidx,:], 'steelblue')

fig2.savefig(figName2)

#------------------------------------------------------------------------------
# Plot instanteneous spec err and cumulative spec err
errK_t = np.mean(((np.abs(dns.Ek_ktt[:,1:M//2] - les.Ek_ktt[:,1:M//2])/les.Ek_ktt[:,1:M//2]))**2, axis=1)
errK = np.cumsum(errK_t)/np.arange(1, len(errK_t)+1)

fig, axs = plt.subplots(1,2, sharex='col', sharey='col')
print(axs.shape)
axs[0].plot(dns.tt, errK_t, 'r:')
axs[1].plot(dns.tt, errK, 'r-')
axs[0].set_yscale('log')
axs[1].set_yscale('log')
#ax.set_ylim([1e-4,1e1])
print("Plot spec errs")
fig.savefig('spec_err.png')
