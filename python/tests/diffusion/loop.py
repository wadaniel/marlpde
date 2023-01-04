#!/bin/python3
import sys
sys.path.append('./../../_model/')

import argparse
from Diffusion import *
 
parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Gridpoints', required=False, type=int, default=32)
parser.add_argument('--dt', help='Timediscretization of URG', required=False, type=float, default=0.01)
parser.add_argument('--tend', help='Length of simulation', required=False, type=float, default=1)
parser.add_argument('--ic', help='Initial condition', required=False, type=str, default='gaussian')
parser.add_argument('--seed', help='Random seed', required=False, type=int, default=42)
parser.add_argument('--episodelength', help='Actual length of episode / number of actions', required=False, type=int, default=100)

args = parser.parse_args()
 
# dns defaults
N    = 512
L    = 2*np.pi
#dt   = 0.01
tEnd = args.tend
nu   = 0.1
ic   = args.ic
seed = args.seed
noise = 0.1

# les & rl defaults
episodeLength = args.episodelength

# DNS baseline
print("Setting up DNS..")
dns = Diffusion(L=L, N=N, dt=args.dt, nu=nu, tend=tEnd, case=ic, noise=0., seed=seed, implicit=True)
dns.simulate()

# Initialize LES
dt_sgs = args.dt
les = Diffusion(L=L, N=N, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
les.setGroundTruth(dns.tt, dns.x, dns.uu)

## run controlled simulation
step = 0
error = 0
nIntermediate = int(tEnd / dt_sgs / episodeLength)
assert nIntermediate > 0
cumreward = 0.

while step < episodeLength and error == 0:
    
    # apply action and advance environment
    actions = [-2.]

    # reweighting
    actions = np.array(actions)
    actions = actions - sum(actions)
 
    try:
        for _ in range(nIntermediate):
            les.step(actions)

    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break
    
    idx = les.ioutnum
    res = les.mapGroundTruth()
    reward = np.mean((res[-1,:] - les.uu[les.ioutnum,:])**2)
    print(reward)

    
    if (np.isnan(reward)):
        print("Nan reward detected")
        error = 1
        break
    
    step += 1
    cumreward += reward

print(cumreward)
