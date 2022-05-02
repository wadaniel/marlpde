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

from KS_clos import *

#------------------------------------------------------------------------------
# les defaults
N    = 32
L    = 22/(2*np.pi)
nu   = 1.0
dt   = 0.1
tTransient = 50
tEnd = 550
tSim = tEnd - tTransient
nSimSteps = int(tSim/dt)
seed = 42
C = 0.1

#------------------------------------------------------------------------------
## transient
les = KS_clos(L=L, N=N, dt=dt, nu=nu, tend=tTransient, seed=seed, ssm=False, dsm=True)
les.simulate(C)
les.fou2real()
#
# #------------------------------------------------------------------------------
# ## restart
v_restart = les.vv[-1,:].copy()
u_restart = les.uu[-1,:].copy()
#
# # set IC
les.IC( u0 = u_restart)
#
# # continue simulation
les.simulate( nsteps=int(tSim/dt), restart = True)
#
# # convert to physical space
# les.fou2real()
#
# # compute energies
# les.compute_Ek()
#
# #------------------------------------------------------------------------------
# ## plot result
# u = les.uu
# e_t = les.Ek_t
# e_tt = les.Ek_tt
# e_ktt = les.Ek_ktt

# k = les.k[:N//2]
#
# fig, axs = plt.subplots(1,3, figsize=(15,15))
# s, n = np.meshgrid(2*np.pi*L/N*(np.array(range(N))+1), les.tt)
#
# axs[0].contourf(s, n, u, 50)
# axs[1].plot(les.tt, e_t)
# axs[1].plot(les.tt, e_tt)
# axs[2].plot(k, 2.0/N * np.abs(e_ktt[0,0:N//2]),'b:')
# axs[2].plot(k, 2.0/N * np.abs(e_ktt[nSimSteps//2,0:N//2]),'b--')
# axs[2].plot(k, 2.0/N * np.abs(e_ktt[-1,0:N//2]),'b')
# axs[2].set_xscale('log')
# axs[2].set_yscale('log')
#
# fig.savefig('simulate_clos.png')
