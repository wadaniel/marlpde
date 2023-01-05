#!/bin/python3

"""
This scripts simulates the Diffusion equation on a grid (N) until t=tEnd. The 
initial condition is set to be approx k^-5/3.
"""

# Discretization grid
N1 = 512
N2 = 32

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from Diffusion import *
from diffusion_environment import setup_dns_default

#------------------------------------------------------------------------------
## set parameters and initialize simulation
numAgents = 32
L    = 2*np.pi
dt   = 0.01
tEnd = 1
nu   = 0.5
implicit = False
seed = 1234
ic   = "sinus"
noise = 0.


dns  = setup_dns_default(ic, N1, dt, nu, tEnd, seed=seed)
#sgs  = setup_dns_default(N2, dt, nu, tEnd, seed=seed)
sgs = Diffusion(L=L, N=N2, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed) 
sgs.setGroundTruth(dns.tt, dns.x, dns.uu)

#------------------------------------------------------------------------------
print("Simulate..")
## simulate
#sgs.simulate()
step = 0
while step < 100:
    # apply action and advance environment
    if numAgents == 1:
        actions = [-2.]
    else:
        actions = [[-2.]] * numAgents
    sgs.step(actions, numAgents=numAgents)
    #sgs.step(None)

    step += 1
 
#------------------------------------------------------------------------------
## plot

print("Plotting evolution.png ...")
fig, axs = plt.subplots(1,1)

cs0 = axs.contourf(dns.tt, dns.x, dns.uu.T, 50, cmap=plt.get_cmap("seismic"))

plt.colorbar(cs0, ax=axs)

fig.savefig('evolution.png')
plt.close()

print("Plotting diffusion_evolution.png ...")
fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
for i in range(16):
    t = i * tEnd / 16
    tidx = int(t / dt)

    l = i % 4
    k = int(i / 4)
 
    axs[k,l].plot(sgs.x, sgs.uu[tidx,:], 'b')
    axs[k,l].plot(dns.x, dns.uu[tidx,:], 'r')

    if ic == "sinus":
        sol = sgs.getAnalyticalSolution(t)
        axs[k,l].plot(sgs.x, sol, '--k')


res = sgs.mapGroundTruth()
mse = np.sum(np.mean((res - sgs.uu)**2, axis=1))/N2
print(f"mse {mse}")
fig.savefig('diffusion_evolution.png'.format())
plt.close()
