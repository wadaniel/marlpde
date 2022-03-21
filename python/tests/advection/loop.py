#!/bin/python3
import sys
sys.path.append('./../../_model/')

import argparse
from Advection import *
 
parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='box')
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)

args = parser.parse_args()
 
# dns defaults
L    = 2*np.pi
dt   = 0.001
tEnd = 10
nu   = 1.
ic   = args.ic

# action defaults
basis = 'hat'
numActions = 1

# les & rl defaults
gridSize = args.N
episodeLength = args.episodelength

# reward defaults
rewardFactor = 10.

# Initialize LES
les = Advection(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, case=ic, noisy=True)
les.setup_basis(numActions, basis)

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
    
    idx = les.ioutnum
    solution = les.getAnalyticalSolution(les.t)
    uDiffMse = ((solution - les.uu[idx,:])**2).mean()
    
    # calculate reward from energy
    reward = -rewardFactor*uDiffMse
    cumreward += reward

    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break
    
    step += 1

print(cumreward)
