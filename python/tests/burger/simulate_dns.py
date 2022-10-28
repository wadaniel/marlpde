#!/bin/python3

"""
This scripts simulates the Burger on two grids (N, N2) up to t=tEnd. We plot the Burger and
the instanteneous energy plus the time averaged energy, and the energy spectra at
start, mid and end of the simulation.
"""

import math

# Discretization grid
N = 512
N_ = 32

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from scipy import interpolate
from scipy.fftpack import fft
from scipy.fftpack import fftfreq


from Burger2 import *
from Burger import*

from scipy import interpolate
from scipy.stats import gaussian_kde
#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.02
nt   = int(tEnd/dt)
ic   = 'turbulence'
seed = 42

dns = Burger2(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed)
urs = Burger2(L=L, N=N_, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed)
#------------------------------------------------------------------------------
print("Simulate dns")
## simulate dns in transient phase and produce IC
dns.simulate()
urs.simulate()
# convert to physical space
dns.fou2real()
# compute energies
dns.compute_Ek()
# filter solution u
dns.filter_u()
# calculate residual field
dns.diff_u()
# compute ground truth sgs
dns.compute_Sgs()

#------------------------------------------------------------------------------
# plot results

# plot contours ---------------------------------------------------------------
fig, axs = plt.subplots(1,2, sharex='col', sharey='col')
axs[0].contourf(dns.x, dns.tt, dns.uu, 100)
axs[1].contourf(urs.x, urs.tt, urs.uu, 100)
axs[0].set_xlabel('x')
axs[0].set_ylabel('t')

print("Plot contour.png")
fig.savefig('contour.png')

# plot evolution ---------------------------------------------------------------

fig2, axs = plt.subplots(4,4, sharex=True, sharey=False, figsize=(15,15))
for i in range(16):
    t = i * tEnd / 16
    tidx = int(t/dt)
    k = int(i / 4)
    l = i % 4

    axs[k,l].plot(dns.x, dns.uu[tidx,:], '--k')
    axs[k,l].plot(urs.x, urs.uu[tidx,:], '--b')
    axs[k, l].set_xlabel('t = {}/16T'.format(i))

figName2 = 'evolution.png'
print("Plotting {} ...".format(figName2))
fig2.savefig(figName2)

# plot sgs histogram -----------------------------------------------------------

print(np.mean(dns.sgs.flatten()))
print(np.std(dns.sgs.flatten()))

smax = dns.sgs.max()
smin = dns.sgs.min()
slevels = np.linspace(smin, smax, 50)
svals = np.linspace(smin, smax, 500)

fig3, axs3 = plt.subplots(1, 3, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(10,10))

axs3[0].contourf(dns.x, dns.tt, dns.sgs)

dnsDensity = gaussian_kde(dns.sgs.flatten())
dnsDensityVals = dnsDensity(svals)
axs3[1].plot(svals, dnsDensityVals)
axs3[1].set_ylim([1e-5, 10])
axs3[1].set_yscale('log')

sfac = 3
dnsMean = np.mean(dns.sgs)
dnsSdev = np.std(dns.sgs)
svals2  = np.linspace(dnsMean-sfac*dnsSdev,dnsMean+sfac*dnsSdev,500)
axs3[2].plot(svals2, dnsDensity(svals2))
axs3[2].set_ylim([1e-5, 10])
axs3[2].set_yscale('log')


figName3 = 'sgs.png'
print("Plotting {} ...".format(figName3))
fig3.savefig(figName3)

# # plot energies ---------------------------------------------------------------
#
# fig, axs = plt.subplots(1,2, sharex='col', sharey='col')
#
# axs[0].plot(dns.tt, dns.Ek_t)
# axs[0].plot(dns.tt, dns.Ek_tt)
#
# k = dns.k[:N//2]
#
# axs[1].plot(k, np.abs(dns.Ek_ktt[0,0:N//2]),'b:')
# axs[1].plot(k, np.abs(dns.Ek_ktt[nt//2,0:N//2]),'b--')
# axs[1].plot(k, np.abs(dns.Ek_ktt[-1,0:N//2]),'b')
# axs[1].plot(k[2:-10], 1e-5*k[2:-10]**(-2),'k--', linewidth=0.5)
#
# axs[0].set_xscale('log')
# axs[1].set_yscale('log')
#
# print("Print dns_energies.png")
# fig.savefig('dns_energies.png')
