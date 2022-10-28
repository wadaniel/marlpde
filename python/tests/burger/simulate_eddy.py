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
ic   = 'turbulence'
seed = 42

urs = Burger2(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed, ssm=True)
#------------------------------------------------------------------------------
print("Simulate urs")
## simulate urs in transient phase and produce IC
urs.simulate()
# convert to physical space
urs.fou2real()
# compute energies
urs.compute_Ek()

#------------------------------------------------------------------------------
# plot results

# plot sgs histogram -----------------------------------------------------------

print(np.mean(urs.sgs.flatten()))
print(np.std(urs.sgs.flatten()))

smax = urs.sgs.max()
smin = urs.sgs.min()
slevels = np.linspace(smin, smax, 50)
svals = np.linspace(smin, smax, 500)

fig3, axs3 = plt.subplots(1, 3, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(10,10))

axs3[0].contourf(urs.x, urs.tt, urs.sgs)

ursDensity = gaussian_kde(urs.sgs.flatten())
ursDensityVals = ursDensity(svals)
axs3[1].plot(svals, ursDensityVals)
axs3[1].set_ylim([1e-5, 100])
axs3[1].set_yscale('log')

sfac = 3
ursMean = np.mean(urs.sgs)
ursSdev = np.std(urs.sgs)
svals2  = np.linspace(ursMean-sfac*ursSdev,ursMean+sfac*ursSdev,500)
axs3[2].plot(svals2, ursDensity(svals2))
axs3[2].set_ylim([1e-5, 100])
axs3[2].set_yscale('log')


figName3 = 'sgs.png'
print("Plotting {} ...".format(figName3))
fig3.savefig(figName3)

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
