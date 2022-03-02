#!/bin/python3
import sys
sys.path.append('./../../_model/')

from Advection import *
 
# dns defaults
L    = 2*np.pi
dt   = 0.01
tEnd = 10
nu   = 1.
ic   = 'box'

# action defaults
basis = 'uniform'
numActions = 1

# les & rl defaults
gridSize = 32
episodeLength = 500

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
    print(solution)
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
