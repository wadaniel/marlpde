#!/bin/python3

""" Script to test absis function """

import sys
sys.path.append('./../../_model/')

from Burger_jax import *
from plotting import *
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='sinus')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
args = parser.parse_args()

N    = 512
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.02
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
rewardFactor = 1 if spectralReward else 1.

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
sgs = Burger_jax(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd)
if spectralReward:
    v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
    sgs.IC( v0 = v0 * gridSize / N )
else:
    sgs.IC( u0 = f_restart(sgs.x) )
sgs.setup_basis(numActions, basis)
sgs.setGroundTruth(dns.tt, dns.x, dns.uu)
## run controlled simulation
error = 0
step = 0
nIntermediate = int(tEnd / dt / episodeLength)
reward = 0.
prevkMseLogErr = 0.
kMseLogErr = 0.
cumreward = 0.
while step < episodeLength and error == 0:

    # apply action and advance environment
    #actions = jnp.asarray(np.random.normal(loc=0., scale=1e-2, size=numActions))
    actions = jnp.ones(numActions)
    print(step)
    #try:
    sgs.step(actions, nIntermediate)
    sgs.compute_Ek()
    #except Exception as e:
    #    print("Exception occured:")
    #    print(str(e))
    #    error = 1
    #    break

    # calculate reward
    if spectralReward:
        #kMseLogErr = np.mean((np.log(dns.Ek_kt[sgs.ioutnum,:gridSize]) - np.log(sgs.Ek_kt[sgs.ioutnum,:gridSize]))**2)
        #reward = -rewardFactor*kMseLogErr
        kMseLogErr = np.mean((np.log(dns.Ek_ktt[sgs.ioutnum,:gridSize]) - np.log(sgs.Ek_ktt[sgs.ioutnum,:gridSize]))**2)
        reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
        prevkMseLogErr = kMseLogErr

    else:
        reward = rewardFactor*sgs.getMseReward()

    cumreward += reward
    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break

    step += 1
print(jnp.shape(sgs.gradient))
print(sgs.gradient)
print(cumreward)

makePlot(dns, sgs, sgs, "grads")
