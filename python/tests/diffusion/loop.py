#!/bin/python3
import sys
sys.path.append('./../../_model/')

import argparse
from Diffusion import *
 
parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Gridpoints', required=False, type=int, default=32)
parser.add_argument('--dt', help='Timediscretization of URG', required=False, type=float, default=0.001)
parser.add_argument('--tend', help='Length of simulation', required=False, type=float, default=5)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='box')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)

args = parser.parse_args()
 
# dns defaults
N    = args.N
L    = 2*np.pi
dt   = 0.001
tEnd = args.tend
nu   = 0.1
ic   = args.ic
seed = args.seed
noise = 0.1

# reward defaults
rewardFactor = 1e6

# action defaults
basis = 'hat'
numActions = 1

# les & rl defaults
episodeLength = args.episodelength

# DNS baseline
print("Setting up DNS..")
dns = Diffusion(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
dns.simulate()

## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

# Initialize LES
dt_sgs = args.dt
les = Diffusion(L=L, N=N, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=0., seed=seed)
les.IC(u0 = f_restart(les.x))
les.setup_basis(numActions, basis)
les.setGroundTruth(dns.tt, dns.x, dns.uu)

## run controlled simulation
error = 0
step = 0
nIntermediate = int(tEnd / dt_sgs / episodeLength)
assert nIntermediate > 0
cumreward = 0.
while step < episodeLength and error == 0:
    
    # apply action and advance environment
    actions = [0.]
    try:
        for _ in range(nIntermediate):
            les.step(actions)
    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break
    
    idx = les.ioutnum
    uTruthToCoarse = les.mapGroundTruth()
    uDiffMse = ((uTruthToCoarse[idx,:] - les.uu[idx,:])**2).mean()
    
    # calculate reward from energy
    # reward = -rewardFactor*(np.abs(les.Ek_tt[step*nIntermediate]-dns.Ek_tt[step*nIntermediate]))
    reward = -rewardFactor*uDiffMse
    cumreward += reward

    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break
    
    #print(step)
    #print(reward)
    step += 1

print(cumreward)
