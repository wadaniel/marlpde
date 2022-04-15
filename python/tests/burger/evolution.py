#!/bin/python3

"""
This scripts simulates the Diffusion equation on a grid (N) until t=tEnd. The 
initial condition is set to be approx k^-5/3.
"""

# Discretization grid
N = 512
N2 = 32

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from plotting import *
from Burger import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L       = 2*np.pi
dt      = 0.001
tEnd    = 5
nu      = 0.005
ic      = 'turbulence'
noise   = 0.01
seed    = 42

dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=True)
sgs0 = Burger(L=L, N=N2, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=True)
sgs = Burger(L=L, N=N2, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=True)

v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:]))
sgs0.IC( v0 = v0 * N2 / N )
sgs0.randfac = dns.randfac

sgs.IC( v0 = v0 * N2 / N )
sgs.randfac = dns.randfac


#------------------------------------------------------------------------------
print("Simulate DNS ..")
## simulate
dns.simulate()
dns.fou2real()
dns.compute_Ek()
print("Compute SGS ..")
dns.compute_Sgs(N2)

print("Simulate SGS..")
## simulate
sgs0.simulate()
# convert to physical space
sgs0.fou2real()
sgs0.compute_Ek()


print("Simulate SGS ..")
sgs.simulate()
sgs.fou2real()
sgs.compute_Ek()
#------------------------------------------------------------------------------
## plot

makePlot(dns, sgs0, sgs, "evolution", False)
