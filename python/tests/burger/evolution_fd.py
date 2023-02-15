#!/bin/python3

# Discretization grid
N = 1024
N2 = 256

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from plotting import *
from Burger import *
from Burger_fd import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L       = 2*pi
dt      = 0.001
s       = 1
tEnd    = 10
ic      = 'turbulence'
forcing = False

ssm = False
dsm = False

#L       = 100
#dt      = 0.01
#s       = 20
#tEnd    = 5000
#ic      = 'sinus'
#forcing = True

nu      = 0.02
#ic      = 'zero'
#ic      = 'turbulence'
#ic      = 'forced'
noise   = 0.
seed    = 42
#forcing = False

dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)
dns.simulate()
dns.compute_Ek()
plotField([dns])
plotAvgSpectrum([dns])

u_restart = dns.uu[0,:].copy()
f_restart = interpolate.interp1d(dns.x, u_restart)

sgs = Burger_fd(L=L, N=N2, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s, ssm=ssm, dsm=dsm)

u0 = f_restart(sgs.x)
sgs.IC( u0 = u0)

sgs.setGroundTruth(dns.x, dns.tt, dns.uu)
sgs.randfac1 = dns.randfac1
sgs.randfac2 = dns.randfac2

#------------------------------------------------------------------------------
print("Simulate DNS ..")
## simulate
#print("Compute SGS ..")
#dns.compute_Sgs(N2)

print("Simulate SGS..")
## simulate
sgs.simulate()
# convert to physical space
sgs.compute_Ek()

#------------------------------------------------------------------------------
## plot
plotField([dns, sgs])
plotAvgSpectrum([dns, sgs])
plotError(dns, sgs)
#makePlot(dns, sgs, sgs, "evolution", False)
