#!/bin/python3

"""
This scripts simulates the Diffusion equation on a grid (N) until t=tEnd. The 
initial condition is set to be approx k^-5/3.
"""

# Discretization grid
N = 512

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
nu      = 0.02
ic      = 'turbulence'
noise   = 0.0
seed    = 42

dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)

#------------------------------------------------------------------------------
print("Simulate..")
## simulate
dns.simulate()
# convert to physical space
dns.fou2real()
dns.compute_Ek()

#------------------------------------------------------------------------------
## plot

makePlot(dns, dns, dns, "evolution", False)
