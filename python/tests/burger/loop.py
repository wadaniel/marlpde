#!/bin/python3
import sys
sys.path.append('./../../_model/')

import argparse
from Burger import *
from plotting import *
 
parser = argparse.ArgumentParser()
parser.add_argument('--gridSize', help='Discretization / number of grid points', required=False, type=int, default=32)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='sinus')
parser.add_argument('--dt', help='Time step', required=False, type=float, default=0.001)
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
parser.add_argument('--dsm', help='Run with dynamic Smagorinsky model', action='store_true')
parser.add_argument('--ssm', help='Run with dynamic Smagorinsky model', action='store_true')

args = parser.parse_args()

# dns defaults
N    = 512
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.02
noise = 0.
ic   = args.ic
seed = args.seed
forcing = False
dforce = False

# action defaults
basis = 'hat'
numActions = 1 #args.gridSize

# sgs & rl defaults
gridSize = args.gridSize
episodeLength = args.episodelength

# reward structure
spectralReward = False

# reward defaults
rewardFactor = 1. if spectralReward else 1.


# DNS baseline
print("Setting up DNS..")
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, forcing=forcing, dforce=dforce, noise=noise, seed=seed)
dns.simulate()
dns.compute_Ek()

# calcuate energies
tAvgEnergy = dns.Ek_tt
print("Done!")

# Initialize LES
sgs = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, case=ic, forcing=forcing, dforce=dforce, noise=0., ssm=args.ssm, dsm=args.dsm)
sgs.randfac1 = dns.randfac1
sgs.randfac2 = dns.randfac2

sgs.setup_basis(numActions, basis)
sgs.setGroundTruth(dns.x, dns.tt, dns.uu)

r = 2.
newx = sgs.x + r
newx[newx>=L] = newx[newx>=L] - L
newx[newx<=0] = newx[newx<=0] + L

if spectralReward:
    v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
    sgs.IC( v0 = v0 * gridSize / dns.N )
else:

    print(newx)
    midx = np.argmax(newx)
    ic = np.concatenate(((sgs.f_truth(newx[:midx+1], 0.)), sgs.f_truth(newx[midx+1:], 0.)))
    print(ic)
    sgs.IC( u0 = ic )
 
## run controlled simulation
error = 0
step = 0
nIntermediate = int(tEnd / dt / episodeLength)

prevkMseLogErr = 0.
kMseLogErr = 0.
cumreward = 0.
while step < episodeLength and error == 0:
    
    reward = 0.
    # apply action and advance environment
    actions = np.zeros(numActions)
    try:
        for _ in range(nIntermediate):
            sgs.step(actions)

            if spectralReward == False:
                reward += rewardFactor*sgs.getMseReward(r)

        sgs.compute_Ek()
    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break
    
    # calculate reward
    if spectralReward:
        kMseLogErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,:gridSize] - sgs.Ek_ktt[sgs.ioutnum,:gridSize])/dns.Ek_ktt[sgs.ioutnum,:gridSize])**2)
        reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
        prevkMseLogErr = kMseLogErr

    cumreward += reward
    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break
    
    step += 1


print(cumreward)

makePlot(dns, sgs, sgs, "loop")
