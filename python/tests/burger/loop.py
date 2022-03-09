#!/bin/python3
import sys
sys.path.append('./../../_model/')

import argparse
from Burger import *
 
parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='box')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)

args = parser.parse_args()

# dns defaults
N    = 512
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.01
ic   = args.ic
seed = args.seed

# action defaults
basis = 'hat'
numActions = 1

# les & rl defaults
gridSize = args.N
episodeLength = args.episodelength

# reward structure
spectralReward = True

# reward defaults
rewardFactor = 0.001 if spectralReward else 1.


# DNS baseline
print("Setting up DNS..")
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noisy=True, seed=seed)
dns.simulate()
dns.fou2real()
dns.compute_Ek()

## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

# calcuate energies
tAvgEnergy = dns.Ek_tt
print("Done!")

# Initialize LES
les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noisy=False)

# Initialize LES
les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noisy=False)
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
    actions = [0.]
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
        #kMseErr = np.mean((dns.Ek_ktt[les.ioutnum,:gridSize] - les.Ek_ktt[les.ioutnum,:gridSize])**2)
        kMseLogErr = np.mean((np.log(dns.Ek_ktt[les.ioutnum,:gridSize]) - np.log(les.Ek_ktt[les.ioutnum,:gridSize]))**2)
        reward = -rewardFactor*kMseLogErr

    else:
        reward = rewardFactor*les.getMseReward()

    cumreward += reward
    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break
    
    step += 1

print(cumreward)
