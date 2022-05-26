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
N    = 1024
L    = 2*np.pi
nu   = 100
dt   = 3*1e-6
tTransient = 1*1e-2
tEnd = 6*1e-3 + tTransient
tSim = tEnd - tTransient
# dt   = 0.01
# tSim = 100
nSteps = int(tSim/dt)

M  = 64
Cs = +3.373e-08

#------------------------------------------------------------------------------
## transient
print("simulate transient")

dns = Kuramoto(L=L, N=N, dt=dt, nu=nu, tend=tTransient)
dns.simulate()
dns.fou2real()
dns.compute_Ek()

#------------------------------------------------------------------------------
## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()

print("simulate DNS")

dns.IC( v0 = v_restart )
dns.simulate( nsteps=int(tSim/dt), restart = True)
dns.fou2real()
dns.compute_Ek()

#------------------------------------------------------------------------------
## les
v0 = np.concatenate((v_restart[:((M+1)//2)], v_restart[-(M-1)//2:])) * M / N

print("Simulate LES")

les = Kuramoto(L=L, N = M, dt=dt, nu=nu, tend=tSim, ssm=True)
les.IC( v0 = v0 )
les.simulate( nsteps=int(tSim/dt), restart=True, Cs=Cs )
print(les.u)
les.fou2real()
les.compute_Ek()

dsm = Kuramoto(L=L, N = M, dt=dt, nu=nu, tend=tSim, dsm=True)
dsm.IC( v0 = v0 )
dsm.simulate( nsteps=int(tSim/dt), restart=True)
print(dsm.u)
dsm.fou2real()
dsm.compute_Ek()

#------------------------------------------------------------------------------

# plot result
u = dns.uu
e_t = dns.Ek_t
e_tt = dns.Ek_tt
e_ktt = dns.Ek_ktt

u_ = les.uu
e_t_ = les.Ek_t
e_tt_ = les.Ek_tt
e_ktt_ = les.Ek_ktt

u__ = dsm.uu
e_t__ = dsm.Ek_t
e_tt__ = dsm.Ek_tt
e_ktt__ = dsm.Ek_ktt

k  = dns.k[:N//2]
k_ = les.k[:M//2]

fig, axs = plt.subplots(2, 3, figsize=(15,15))

print(dns.tt.shape)
print(dns.x.shape)
print(dns.uu.shape)

print(les.tt.shape)
print(les.x.shape)
print(les.uu.shape)

axs[0, 0].contourf(dns.x, dns.tt, dns.uu)
axs[1, 0].plot(k, 2.0/N * np.abs(e_ktt[0,0:N//2]),'b:')
axs[1, 0].plot(k, 2.0/N * np.abs(e_ktt[nSteps//2,0:N//2]),'b--')
axs[1, 0].plot(k, 2.0/N * np.abs(e_ktt[-1,0:N//2]),'b')
axs[1, 0].set_xscale('log')
axs[1, 0].set_yscale('log')

axs[0, 1].contourf(les.x, les.tt, les.uu)
axs[1, 1].plot(k_, 2.0/M * np.abs(e_ktt_[0,0:M//2]),'b:')
axs[1, 1].plot(k_, 2.0/M * np.abs(e_ktt_[nSteps//2,0:M//2]),'b--')
axs[1, 1].plot(k_, 2.0/M * np.abs(e_ktt_[-1,0:M//2]),'b')
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')

axs[0, 2].contourf(dsm.x, dsm.tt, dsm.uu)
axs[1, 2].plot(k_, 2.0/M * np.abs(e_ktt__[0,0:M//2]),'b:')
axs[1, 2].plot(k_, 2.0/M * np.abs(e_ktt__[nSteps//2,0:M//2]),'b--')
axs[1, 2].plot(k_, 2.0/M * np.abs(e_ktt__[-1,0:M//2]),'b')
axs[1, 2].set_xscale('log')
axs[1, 2].set_yscale('log')

print("Plotting simulateKS.png")
fig.savefig('simulateKS.png')
