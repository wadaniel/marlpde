#!/bin/python3
import sys
sys.path.append('./../../_model/')

import argparse
from Burger import *
from Burger_fd import *
from plotting import *
 
parser = argparse.ArgumentParser()
parser.add_argument('--N', help='DNS gridsize', required=False, type=int, default=1024)
parser.add_argument('--gridSize', help='Discretization / number of grid points', required=False, type=int, default=256)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='turbulence')
parser.add_argument('--L', help='Domain width', required=False, type=float, default=2*np.pi)
parser.add_argument('--T', help='Simulation length', required=False, type=float, default=5)
parser.add_argument('--dt', help='Time step', required=False, type=float, default=0.001)
parser.add_argument('--stepper', help='Time multiplicator SGS', required=False, type=int, default=1)
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--forcing', help='Random forcing', action='store_true')
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=500)
parser.add_argument('--dsm', help='Run with dynamic Smagorinsky model', action='store_true')
parser.add_argument('--ssm', help='Run with dynamic Smagorinsky model', action='store_true')

args = parser.parse_args()

# dns defaults
N    = args.N
L    = args.L
dt   = args.dt
tEnd = args.T
ic   = args.ic
seed = args.seed
forcing = args.forcing
dforce = True
ssmforce = True
nu   = 0.02
noise = 0.
offset = 0.

# reward structure
spectralReward = True

# action defaults
basis = 'hat'
numActions = 1 #args.gridSize

# sgs & rl defaults
gridSize = args.gridSize
episodeLength = args.episodelength
stepper = args.stepper

# reward defaults
rewardFactor = 1. if spectralReward else 1.

# DNS baseline
print("Setting up DNS..")
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, forcing=forcing, noise=0., seed=seed, s=args.stepper)
dns.simulate()
dns.compute_Ek()

# calcuate energies
tAvgEnergy = dns.Ek_tt
print("Done!")

# Initialize LES
print("Setting up SGS..")
 #Initialize LES
sgs = Burger_fd(L=L, 
        N=gridSize, 
        dt=dt, 
        nu=nu, 
        tend=tEnd, 
        case=ic, 
        forcing=forcing, 
        dforce=dforce, 
        ssmforce=ssmforce,
        noise=0., 
        s=stepper,
        ssm=args.ssm,
        dsm=args.dsm)

## copy random numbers
sgs.randfac1 = dns.randfac1
sgs.randfac2 = dns.randfac2

sgs.setup_basis(numActions, basis)
sgs.setGroundTruth(dns.x, dns.tt, dns.uu)

newx = sgs.x + sgs.offset
newx[newx>L] = newx[newx>L] - L
newx[newx<0] = newx[newx<0] + L

midx = np.argmax(newx)
if midx == len(newx)-1:
    ic = sgs.f_truth(newx, 0)
else:
    ic = np.concatenate(((sgs.f_truth(newx[:midx+1], 0.)), sgs.f_truth(newx[midx+1:], 0.)))
sgs.IC( u0 = ic )

## run controlled simulation
error = 0
step = 0
kPrevRelErr = 0.
nIntermediate = int(tEnd / (stepper*dt) / episodeLength)
assert nIntermediate > 0, print(f"{nIntermediate}, {tEnd}, {stepper}, {dt}, {episodeLength}")

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
        sgs.compute_Ek()
        kRelErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,1:gridSize//2] - sgs.Ek_ktt[sgs.ioutnum,1:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,1:gridSize//2])**2)
        reward = rewardFactor*(kPrevRelErr-kRelErr)
        kPrevRelErr = kRelErr

    cumreward += reward
    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break
    
    step += 1


print(cumreward)

makePlot(dns, sgs, sgs, "loop")
