#!/bin/python3

"""
"""

import math

# Discretization grid
N = 32

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
ic   = 'sinus'
seed = 42

urs_s = Burger2(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed, ssm=True, forcing=True)
urs_d = Burger2(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed, dsm=True, forcing=True)
#------------------------------------------------------------------------------
print("Simulate urs")
## simulate urs in transient phase and produce IC
urs_s.simulate()
urs_d.simulate()
# convert to physical space
urs_s.fou2real()
urs_d.fou2real()
# compute energies
urs_s.compute_Ek()
urs_d.fou2real()

#------------------------------------------------------------------------------
# plot results

# plot sgs histogram -----------------------------------------------------------

print(np.mean(urs_s.sgs.flatten()))
print(np.std(urs_s.sgs.flatten()))

smax = urs_s.sgs.max()
smin = urs_s.sgs.min()
slevels = np.linspace(smin, smax, 50)
svals = np.linspace(smin, smax, 500)

fig3, axs3 = plt.subplots(1, 3, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(10,10))

axs3[0].contourf(urs_s.x, urs_s.tt, urs_s.sgs)

urs_sDensity = gaussian_kde(urs_s.sgs.flatten())
urs_sDensityVals = urs_sDensity(svals)
axs3[1].plot(svals, urs_sDensityVals)
axs3[1].set_ylim([1e-5, 100])
axs3[1].set_yscale('log')

sfac = 3
urs_sMean = np.mean(urs_s.sgs)
print("static mean")
print(urs_sMean)
urs_sSdev = np.std(urs_s.sgs)
print("static deviation")
print(urs_sSdev)
svals2  = np.linspace(urs_sMean-sfac*urs_sSdev,urs_sMean+sfac*urs_sSdev,500)
axs3[2].plot(svals2, urs_sDensity(svals2))
axs3[2].set_ylim([1e-5, 100])
axs3[2].set_yscale('log')

figName3 = 'sgs_ssm.png'
print("Plotting {} ...".format(figName3))
fig3.savefig(figName3)

###########################################################

print(np.mean(urs_d.sgs.flatten()))
print(np.std(urs_d.sgs.flatten()))

smax = urs_d.sgs.max()
smin = urs_d.sgs.min()
slevels = np.linspace(smin, smax, 50)
svals = np.linspace(smin, smax, 500)

fig4, axs4 = plt.subplots(1, 3, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(10,10))

axs4[0].contourf(urs_d.x, urs_d.tt, urs_d.sgs)

urs_dDensity = gaussian_kde(urs_d.sgs.flatten())
urs_dDensityVals = urs_dDensity(svals)
axs4[1].plot(svals, urs_dDensityVals)
axs4[1].set_ylim([1e-5, 1000])
axs4[1].set_yscale('log')

sfac = 3
urs_dMean = np.mean(urs_d.sgs)
print("dynamic mean")
print(urs_dMean)
urs_dSdev = np.std(urs_d.sgs)
print("dynamic deviation")
print(urs_dSdev)
svals2  = np.linspace(urs_dMean-sfac*urs_dSdev,urs_dMean+sfac*urs_dSdev,500)
axs4[2].plot(svals2, urs_dDensity(svals2))
axs4[2].set_ylim([1e-5, 1000])
axs4[2].set_yscale('log')


figName4 = 'sgs_dsm.png'
print("Plotting {} ...".format(figName4))
fig4.savefig(figName4)

# # plot energies ---------------------------------------------------------------
#
# fig, axs = plt.subplots(1,2, sharex='col', sharey='col')
#
# axs[0].plot(urs.tt, urs.Ek_t)
# axs[0].plot(urs.tt, urs.Ek_tt)
#
# k = urs.k[:N//2]
#
# axs[1].plot(k, np.abs(urs.Ek_ktt[0,0:N//2]),'b:')
# axs[1].plot(k, np.abs(urs.Ek_ktt[nt//2,0:N//2]),'b--')
# axs[1].plot(k, np.abs(urs.Ek_ktt[-1,0:N//2]),'b')
# axs[1].plot(k[2:-10], 1e-5*k[2:-10]**(-2),'k--', linewidth=0.5)
#
# axs[0].set_xscale('log')
# axs[1].set_yscale('log')
#
# print("Print urs_energies.png")
# fig.savefig('urs_energies.png')
