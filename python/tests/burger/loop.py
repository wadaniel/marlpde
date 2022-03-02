#!/bin/python3
import sys
sys.path.append('./../../_model/')

from Burger import *
 
# dns defaults
N    = 512
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.01

# action defaults
basis = 'uniform'
numActions = 1

# les & rl defaults
gridSize = 8
episodeLength = 500

# reward defaults
rewardFactor = 1.

# DNS baseline
print("Setting up DNS..")
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd)
dns.IC(case='box')
dns.simulate()
dns.fou2real()
dns.compute_Ek()

## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

# calcuate energies
tAvgEnergy = dns.Ek_tt
print("Done!")

# Initialize LES
les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noisy=True)
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
    
    step += 1

print(cumreward)
