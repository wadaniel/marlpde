#!/bin/python3

# Discretization grid
N = 1024
N2 = 64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../python/_model/')

import numpy as np
from scipy.stats import pearsonr

from plotting import *
from Burger import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L       = 2*pi
dt      = 0.01
s       = 1
tEnd    = 5
nu      = 0.02
#ic      = 'zero'
#ic      = 'turbulence'
#ic      = 'sinus'
ic      = 'forced'
noise   = 0.
seed    = 42
forcing = False

dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)

#v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:]))
#sgs0.IC( v0 = v0 * N2 / N )

#sgs0.randfac1 = dns.randfac1
#sgs0.randfac2 = dns.randfac2

#sgs.IC( v0 = v0 * N2 / N )
#sgs.randfac1 = dns.randfac1
#sgs.randfac2 = dns.randfac2

#------------------------------------------------------------------------------
print("Simulate DNS ..")

## simulate
dns.simulate()
dns.compute_Ek()
dns.compute_Sgs(N2)

dnsSgs = dns.sgsHistory.flatten()
dnsSgsAlt = dns.sgsHistoryAlt.flatten()
## Compute Static Smagorinsky Terms
cs = 0.01

delta  = 2*np.pi/N
dx  = dns.dx
dx2 = dx**2
um = np.roll(dns.uu, 1)
up = np.roll(dns.uu, -1)
dudx = (dns.u - um)/dx 
d2udx2 = (up - 2*dns.u + um)/dx2
nuSSM = (cs*delta)**2*np.abs(dudx)
sgs = nuSSM*d2udx2

sgs = sgs.flatten()

c1, _ = pearsonr(dnsSgs, sgs)
print(f"Correlation {c1}")
c2, _ = pearsonr(dnsSgsAlt, sgs)
print(f"CorrelationAlt {c2}")

plt.figure(1)
plt.scatter(dnsSgs, sgs, s=1, alpha=0.5)
plt.savefig('correlation.png')

plt.figure(2)
plt.scatter(dnsSgsAlt, sgs, s=1, alpha=0.5)
plt.savefig('correlationAlt.png')
