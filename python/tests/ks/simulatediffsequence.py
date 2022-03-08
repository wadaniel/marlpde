#!/bin/python3

"""
This scripts simulates the KS on two grids (N1, N2) up to t=tEnd. We plot the Burger and
the instanteneous energy plus the time averaged energy, and the energy spectra at 
start, mid and end of the simulation.
"""

import sys
import math
import numpy as np

# Discretization grid
N1 = 2048
N2 = 32
M = int(math.log2(N1 / N2)) + 1
Nx = np.clip(N2*2**np.arange(0., M), a_min=0, a_max=N1).astype(int)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
sys.path.append('./../../_model/')

from scipy import interpolate
from scipy.fft import fftfreq

from KS import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L          = 22
dt         = 0.01
tTransient = 50
tEnd       = 1000
nu         = 1.
tSim       = tEnd-tTransient
dns = KS(L=L, N=N1, dt=dt, nu=nu, tend=tTransient)

#------------------------------------------------------------------------------
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()

#------------------------------------------------------------------------------
## plot result
u = dns.uu
#------------------------------------------------------------------------------
## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()


#------------------------------------------------------------------------------
print("Simulate DNS")
# restart from physical space
dns.IC( u0 = u_restart )

## simulate DNS in transient phase and produce IC
dns.simulate( nsteps=int(tSim/dt), restart=True )
# convert to physical space
dns.fou2real()
# compute energies
dns.compute_Ek()
# IC and interpolation
IC = dns.u0.copy()
f_IC = interpolate.interp1d(dns.x, IC)
f_dns = interpolate.interp2d(dns.x, dns.tt, dns.uu, kind='cubic')

#------------------------------------------------------------------------------
print("plot DNS")
k1 = dns.k[:N1//2]

fig, axs = plt.subplots(M+1, 5, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(15,15))
axs[0,0].contourf(dns.x, dns.tt, dns.uu, 50)

axs[0,2].plot(dns.tt, dns.Ek_t)
axs[0,2].plot(dns.tt, dns.Ek_tt)

nt = int(tSim/dt)
axs[0,4].plot(k1, np.abs(dns.Ek_ktt[0,0:N1//2]),'b:')
axs[0,4].plot(k1, np.abs(dns.Ek_ktt[nt//2,0:N1//2]),'b--')
axs[0,4].plot(k1, np.abs(dns.Ek_ktt[-1,0:N1//2]),'b')
axs[0,4].set_xscale('log')
axs[0,4].set_yscale('log')


#------------------------------------------------------------------------------

idx = 1

for N in Nx:
    print("Simulate SGS (N={})".format(N))
    ## simulate SGS from IC
    sgs = KS(L=L, N=N, dt=dt, nu=nu, tend=tSim)
    u0 = f_IC(sgs.x)
    sgs.IC(u0 = u0)

    sgs.simulate()
    # convert to physical space
    sgs.fou2real()
    # compute energies
    sgs.compute_Ek()

#------------------------------------------------------------------------------

    errEk_t = dns.Ek_t - sgs.Ek_t
    errEk_tt = dns.Ek_tt - sgs.Ek_tt
    errEk_ktt = ((dns.Ek_ktt[:, :N2] - sgs.Ek_ktt[:, :N2])**2).mean(axis=1)
    udns_int = f_dns(sgs.x, sgs.tt)

    errU = np.abs(sgs.uu-udns_int)
#------------------------------------------------------------------------------
  
    k2 = sgs.k[:N//2]

    # Plot solution
    axs[idx,0].contourf(sgs.x, sgs.tt, sgs.uu, 50)
    
    # Plot difference to dns
    axs[idx,1].contourf(sgs.x, sgs.tt, errU, 50)

    # Plot instanteneous energy and time averaged energy
    axs[idx,2].plot(sgs.tt, sgs.Ek_t)
    axs[idx,2].plot(sgs.tt, sgs.Ek_tt)
 
    # Plot energy differences
    axs[idx,3].plot(sgs.tt, errEk_t)
    axs[idx,3].plot(sgs.tt, errEk_tt)
    axs[idx,3].plot(sgs.tt, errEk_ktt)
    axs[idx,3].set_yscale('log')

    # Plot energy spectrum at start, mid and end of simulation
    axs[idx,4].plot(k2, np.abs(sgs.Ek_ktt[0,0:N//2]),'b:')
    axs[idx,4].plot(k2, np.abs(sgs.Ek_ktt[nt//2,0:N//2]),'b--')
    axs[idx,4].plot(k2, np.abs(sgs.Ek_ktt[-1,0:N//2]),'b')
    axs[idx,4].set_xscale('log')
    axs[idx,4].set_yscale('log')

    idx += 1

fig.savefig('simulate_energies_seq.png')
