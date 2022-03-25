#!/bin/python3
import sys
sys.path.append('./../../_model/')

import argparse
from Burger import *
 
parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='box')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--noise', help='Random IC noise', required=False, type=float, default=0.)
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

# reward defaults
#rewardFactor = 0.001 if spectralReward else 1.
rewardFactor = 1


# DNS baseline
print("Setting up DNS..")
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=args.noise, seed=seed)
dns.simulate()
dns.fou2real()
dns.compute_Ek()
print("DONE")

## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

print("Running base0")
base0 = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
base0.IC( u0 = f_restart(base0.x) )
base0.simulate()
base0.fou2real()
print("DONE")

# Initialize LES
les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
les.IC( u0 = f_restart(les.x) )
les.setup_basis(numActions, basis)
les.setGroundTruth(dns.tt, dns.x, dns.uu)

## run controlled simulation
step = 0
error = 0
nIntermediate = int(tEnd / dt / episodeLength)
cumreward = 0.
cumreward0 = 0.
while step < episodeLength and error == 0:
    
    # apply action and advance environment
    actions = [0.01]
  
    vBase = les.vv[les.ioutnum,:]
    uBase = les.uu[les.ioutnum,:]

    try:
        for _ in range(nIntermediate):
            les.step(actions)

        les.compute_Ek()
        les.fou2real()
    except Exception as e:
        print("Exception occured in LES:")
        print(str(e))
        error = 1
        break
   
    try:
        for _ in range(nIntermediate):
            vBase = vBase - dt*0.5*les.k1*fft(uBase**2) + dt*nu*les.k2*vBase
            uBase = np.real(ifft(vBase))
    except Exception as e:
        print("Exception occured in BASE:")
        print(str(e))
        error = 2
        break
  

    # calculate reward
    uTruth = les.mapGroundTruth()
   
    try:
        uLesDiffMse = ((uTruth[les.ioutnum,:] - les.uu[les.ioutnum,:])**2).mean()
        uBaseDiffMse = ((uTruth[les.ioutnum,:] - uBase)**2).mean()
        uBase0DiffMse = ((uTruth[les.ioutnum,:] - base0.uu[les.ioutnum,:])**2).mean()
    except Exception as e:
        print("Exception occured in MSE:")
        print(str(e))
        reward = -np.inf
        error = 1
        break
    
    reward = rewardFactor*(uBaseDiffMse-uLesDiffMse)
    reward0 = rewardFactor*(uBase0DiffMse-uLesDiffMse)
   
    cumreward += reward
    cumreward0 += reward0
    
    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break
  
    if (np.isnan(reward0)):
        print("Nan reward0 detected")
        error = 1
        break
    
    step += 1

print("cumreward")
print(cumreward)
print("cumreward0")
print(cumreward0)
