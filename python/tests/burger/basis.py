#!/bin/python3

""" Script to test absis function """

import sys
sys.path.append('./../../_model/')

from Burger import *
  
N    = 1024
L    = 2*np.pi
dt   = 0.0005
tEnd = 5
nu   = 0.01
basis = 'hat'
gridSize = 32
episodeLength = 500

numActions = 4

# reward defaults
rewardFactor = 1.

# DNS baseline
print("Setting up DNS..")
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd)
dns.simulate()
dns.fou2real()
dns.compute_Ek()


## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

# calcuate energies
tAvgEnergy = dns.Ek_tt
print("Done!")

  
# Initialize LES
les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd)
les.setup_basis(numActions, basis)
les.IC( u0 = f_restart(les.x) )

## run controlled simulation
error = 0
step = 0
nIntermediate = int(tEnd / dt / episodeLength)
cumreward = 0.
while step < episodeLength and error == 0:
    
    # apply action and advance environment
    actions = np.random.normal(loc=0., scale=1e-4, size=numActions)
    try:
        for _ in range(nIntermediate):
            les.step(actions)
        les.compute_Ek()
    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break
    
    # calculate reward from energy
    tAvgEnergyLES = les.Ek_tt
    reward = -rewardFactor*(np.abs(tAvgEnergyLES[step*nIntermediate]-tAvgEnergy[step*nIntermediate]))
    cumreward += reward

    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break
    
    step += 1

print(cumreward)
