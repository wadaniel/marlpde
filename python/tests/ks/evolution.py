#!/bin/python3

"""
This scripts simulates the Diffusion equation on a grid (N) until t=tEnd. The 
initial condition is set to be approx k^-5/3.
"""

# Discretization grid
N    = 2048
N2   = 32

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from plotting import *
from KS import *

# DNS baseline
def setup_dns_default(N, dt, nu , seed):
    print("[ks_environment] setting up default dns")

    # simulate transient period
    dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient, seed=seed)
    dns.simulate()
    dns.fou2real()
    u_restart = dns.uu[-1,:].copy()
    v_restart = dns.vv[-1,:].copy()

    # simulate rest
    dns.IC( u0 = u_restart)
    dns.simulate( nsteps=int(tSim/dt), restart=True )
    dns.fou2real()
    dns.compute_Ek()

    return dns
 
#------------------------------------------------------------------------------
## set parameters and initialize simulation
L    = 100
nu   = 1.0
dt   = 0.25
seed = 42
tTransient = 50
tEnd = 550
tSim = tEnd - tTransient
nSimSteps = tSim/dt
episodeLength = 500


dns = setup_dns_default(N, dt, nu, seed)
v0 = np.concatenate((dns.v0[:((N2+1)//2)], dns.v0[-(N2-1)//2:]))

sgs = KS(L=L, N = N2, dt=dt, nu=nu, tend=tSim)
sgs.IC( v0 = v0 * N2 / N )
sgs.setup_basis(N2)


#------------------------------------------------------------------------------
print("Simulate DNS ..")
## simulate
dns.fou2real()
dns.compute_Ek()
print("Compute SGS ..")
dns.compute_Sgs(N2)

print("Simulate SGS ..")
sgs.simulate()
sgs.fou2real()
sgs.compute_Ek()
#------------------------------------------------------------------------------
## plot

makePlot(dns, sgs, sgs, "evolution", False)
