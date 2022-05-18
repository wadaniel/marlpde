#!/bin/python3

"""
This scripts simulates the KS on a fine grid (N) up to t=tEnd. We plot the KS and
the instanteneous energy plus the time averaged energy, and the energy spectra at
start, mid and end of the simulation.
"""

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from scipy.fft import fftfreq

from Kuramoto import *

#------------------------------------------------------------------------------
# DNS defaults
N    = 512
L    = 32*np.pi
nu   = 1
tSim = 30
dt   = tSim/20000
nt   = int(tSim/dt)
seed = 42
n    = 9

dt_arr  = np.zeros(n)
nt_arr  = np.zeros(n)
err_L2  = np.zeros(n)
err_L1  = np.zeros(n)
err_inf = np.zeros(n)
u_i     = np.zeros([n, N])
#------------------------------------------------------------------------------
## transient
print("Simulate exact")
dns = Kuramoto(L=L, N=N, dt=dt, nu=nu, tend=tSim )
dns.IC( case = 'ETDRK4')
dns.simulate()
u = dns.uu[dns.ioutnum,:]

print("Simulate rest")
for i in range(0,n):
    rel = 2**(i+1)
    dt_i = dt*rel
    print(dt_i)
    dt_arr[i] = dt_i
    nt_arr[i] = tSim//dt_i
    dns_i = Kuramoto(L=L, N=N, dt=dt_i, nu=nu, tend=tSim )
    dns_i.IC( case = 'ETDRK4')
    dns_i.simulate()
    u_i = dns_i.uu[dns_i.ioutnum,:]

    err_L2[i]  = np.linalg.norm((u_i - u)/max(u), ord=2)
    print(err_L2[i])
    err_L1[i]  = np.linalg.norm((u_i - u)/max(u), ord=1)
    err_inf[i] = np.linalg.norm((u_i - u)/max(u), ord=np.inf)

#------------------------------------------------------------------------------
## save result
np.save("verification", (1/nt_arr, err_L2, err_L1, err_inf))

#------------------------------------------------------------------------------
## plot result
fig, axs = plt.subplots(1, 3)

axs[0].plot(1/nt_arr, err_L2 )
axs[1].plot(1/nt_arr, err_L1 )
axs[2].plot(1/nt_arr, err_inf, "o")
x = np.linspace(min(1/nt_arr), max(1/nt_arr), 100)
axs[2].plot(x, 5*10**4*x**4,"--k")


axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[2].set_xscale('log')
axs[2].set_yscale('log')

print(dt_arr)
print(err_L2)
print(np.polyfit(np.log(dt_arr), np.log(err_L2), 1))
print(np.polyfit(np.log(dt_arr), np.log(err_L1), 1))
print(np.polyfit(np.log(dt_arr), np.log(err_inf), 1))

print("Plot orderacc.png")
plt.show()
# fig.savefig('orderacc.png')
# print(dt_arr)
# print(dt_arr)
# print(err_L2)
# print(np.polyfit(np.log(dt_arr), np.log(err_L2), 1))
# print(np.polyfit(np.log(dt_arr), np.log(err_L1), 1))
# print(np.polyfit(np.log(dt_arr), np.log(err_inf), 1))
