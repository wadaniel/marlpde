#!/bin/python3
import sys
sys.path.append('./../../_model/')
from KS import *
 
import matplotlib.pyplot as plt

# LES defaults
numActions = 32
gridSize = 32
 
# DNS defaults
N    = 2048
L    = 22
nu   = 1.0
dt   = 0.25
seed = 42
tTransient = 50
tEnd = 550
tSim = tEnd - tTransient
nSimSteps = tSim/dt
episodeLength = 500

rewardFactor = 1
 
# DNS baseline
def setup_dns_default(N, dt, nu , seed):
    print("[ks_environment] setting up default dns")

    # simulate transient period
    dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient, seed=seed)
    dns.simulate()
    dns.fou2real()
    u_restart = dns.uu[-1,:].copy()
    v_restart = dns.vv[-1,:].copy()

    # simulate rest
    dns.IC( u0 = u_restart)
    dns.simulate( nsteps=int(tSim/dt), restart=True )
    dns.fou2real()
    dns.compute_Ek()

    return dns
 
# DNS baseline
dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient)
dns.simulate()
dns.fou2real()
  
dns = setup_dns_default(N, dt, nu, seed)

## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()

v0 = np.concatenate((v_restart[:((gridSize+1)//2)], v_restart[-(gridSize-1)//2:])) * gridSize / dns.N
 
## create interpolated IC
f_restart = interpolate.interp1d(dns.x, u_restart, kind='cubic')

# init rewards
rewards = []

# Initialize LES
sgs = KS(L=L, N = gridSize, dt=dt, nu=nu, tend=tSim)
sgs.IC( v0 = v0 )
sgs.setup_basis(numActions)
 
## run controlled simulation
error = 0
step = 0
nIntermediate = int(tSim / dt / episodeLength)
prevkMseLogErr = 0.
kMseLogErr = 0.
reward = 0.
cumreward = 0.

while step < episodeLength and error == 0:
        
    # apply action and advance environment
    actions = np.zeros(numActions)

    try:
        for _ in range(nIntermediate):
            sgs.step(actions)
        sgs.compute_Ek()
        sgs.fou2real()

    except Exception as e:
        print("[ks_environment] Exception occured:")
        print(str(e))
        error = 1
        break
    
    # get new state
    state = sgs.getState().flatten().tolist()
    if(np.isnan(state).any() == True):
        print("[ks_environment] Nan state detected")
        error = 1
        break

    kMseLogErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,1:gridSize//2] - sgs.Ek_ktt[sgs.ioutnum,1:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,1:gridSize//2])**2)
    reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
    prevkMseLogErr = kMseLogErr

    cumreward += reward
    
    if (np.isnan(reward)):
        print("[ks_environment] Nan reward detected")
        error = 1
        break

    step += 1

print(cumreward)
