#!/bin/python3

# Discretization grid
N = 32

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
L       = 2*pi
dt      = 0.001
s       = 20
tEnd    = 5
ic      = 'turbulence'
forcing = False

ssm = False
dsm = True
levels=20 #np.linspace(-0.35,0.35,20)
#levels=[-0.36, -0.32, -0.28, -0.24, -0.2, -0.16, -0.12, -0.08, -0.04, 0., 0.04, 0.08, 0.12, 0.16, 0.2 , 0.24, 0.28, 0.32, 0.36]

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

dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s, ssm=ssm, dsm=dsm)
dns.simulate()
dns.compute_Ek()

fig = plt.figure()
ax = fig.add_subplot(111)
cs = ax.contourf(dns.x, dns.tt, dns.sgsHistory, levels)

print(cs.levels)
print(max(dns.sgsHistory.flatten()))
print(min(dns.sgsHistory.flatten()))

ratio=1
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
fig.colorbar(cs)
plt.tight_layout()
plt.savefig(f"dsmagorinsky_{N}_alt.eps")
