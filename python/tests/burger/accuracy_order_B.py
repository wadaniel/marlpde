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

from Burger import *

#------------------------------------------------------------------------------
# DNS defaults
N    = 124
L    = 2*np.pi
nu   = 1
dt   = 1e-6
dt_start = 1e-6
tSim= 1e-1
nt = int(tSim/dt)
seed = 42
n = 12

dt_arr = np.zeros(n)
err_L2   = np.zeros(n)
err_L1   = np.zeros(n)
err_inf   = np.zeros(n)
u_i    = np.zeros([n, N])
#------------------------------------------------------------------------------
## transient
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tSim)

print("Simulate sol")

dns.simulate(nsteps=int(tSim/dt))
dns.fou2real()
u_restart = dns.uu[-1,:].copy()
v_restart = dns.vv[-1,:].copy()
#
#u = dns.uu[dns.ioutnum,:]
u = dns.u

print("Simulate rest")

for i in range(0,n):
    rel = 2**(i+1)
    dt_i = dt*rel
    dt_arr[i] = dt_i
    dns_i = Burger(L=L, N=N, dt=dt_i, nu=nu, tend=tSim)
    dns_i.IC( u0 = u_restart)
    dns_i.simulate( nsteps=int(tSim/dt_i), restart=True )
    #u_i[i] = dns_i.uu[dns_i.ioutnum,:]
    u_i[i] = dns_i.u
    err_L2[i] = np.linalg.norm(u_i[i] - u, 2)
    # err_L1[i] = np.linalg.norm(u[i] - u, 1)
    # err_inf[i] = np.linalg.norm(u[i] - u, np.inf)

#
#------------------------------------------------------------------------------
## plot result

fig, axs = plt.subplots(1, 3)

axs[0].plot(dt_arr, err_L2)
# axs[1].plot(dt_arr, err_L1)
# axs[2].plot(dt_arr, err_inf)

axs[0].set_xscale('log')
axs[0].set_yscale('log')
# axs[1].set_xscale('log')
# axs[1].set_yscale('log')
# axs[2].set_xscale('log')
# axs[2].set_yscale('log')

print("Plot orderacc.png")
fig.savefig('orderacc.png')
print(dt_arr)
print(err_L2)
print(np.polyfit(np.log(dt_arr), np.log(err_L2), 1))
