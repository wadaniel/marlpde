#!/bin/python3

""" Script to test absis function """

import sys
sys.path.append('./../../_model/')

from Burger_jax import *
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='box')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
args = parser.parse_args()

N    = 64
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.01
noise = 0.0
ic   = args.ic
seed = args.seed

# action defaults
basis = 'hat'
numActions = 16

gridSize = N
episodeLength = args.episodelength

# reward structure
spectralReward = True

# reward defaults
#rewardFactor = 0.001 if spectralReward else 1.
rewardFactor = 100 if spectralReward else 1.

# DNS baseline

print("Setting up DNS..")
dns = Burger_jax(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd)
dns.setup_basis(numActions, basis)
dns.simulate()
dns.compute_Ek()


## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

# calcuate energies
tAvgEnergy = dns.Ek_tt
print("Done!")


# Initialize LES
les = Burger_jax(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd)
if spectralReward:
    les.IC( v0 = dns.v0[:gridSize] * gridSize / N )
else:
    les.IC( u0 = f_restart(les.x) )
les.setup_basis(numActions, basis)
les.setGroundTruth(dns.tt, dns.x, dns.uu)
## run controlled simulation
error = 0
step = 0
nIntermediate = int(tEnd / dt / episodeLength)
cumreward = 0.
while step < episodeLength and error == 0:

    # apply action and advance environment
    #actions = jnp.asarray(np.random.normal(loc=0., scale=1e-2, size=numActions))
    actions = jnp.ones(numActions)
    print(step)
    try:
        for _ in range(nIntermediate):
            les.step(actions)
        les.compute_Ek()
    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break

    # calculate reward
    if spectralReward:
        # Time-averaged energy spectrum as a function of wavenumber
        kMseErr = np.mean((dns.Ek_ktt[les.ioutnum,:gridSize] - les.Ek_ktt[les.ioutnum,:gridSize])**2)
        #kMseErr = np.mean((np.log(dns.Ek_ktt[les.ioutnum,:gridSize]) - np.log(les.Ek_ktt[les.ioutnum,:gridSize]))**2)
        reward = -rewardFactor*kMseErr

    else:
        reward = rewardFactor*les.getMseReward()

    cumreward += reward
    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break

    step += 1
print(jnp.shape(les.gradient))
print(les.gradient)
print(cumreward)
