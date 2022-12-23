from Diffusion import *
from plotting import makeDiffusionPlot

import matplotlib.pyplot as plt 

# defaults
NDNS=512
L = 2*np.pi

# basis defaults
basis = 'hat'

def setup_dns_default(ic, NDNS, dt, nu, tend, seed):
    print("[diffusion_environment] Setting up default dns with args ({}, {}, {}, {} )".format(NDNS, dt, nu, seed))
    dns = Diffusion(L=L, N=NDNS, dt=dt, nu=nu, tend=tend, case=ic, noise=0., implicit = True)
    dns.simulate()
    return dns

def environment( s , N, tEnd, dtSgs, nu, episodeLength, ic, noise, seed, dnsDefault, nunoise=False, numAgents=1, version=0):
    
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False

    les = Diffusion(L=L, N=N, dt=dtSgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
    les.setGroundTruth(dnsDefault.tt, dnsDefault.x, dnsDefault.uu)

    step = 0
    error = 0
    cumreward = 0.

    state = les.getState(numAgents)
    s["State"] = state

    while step < episodeLength and error == 0:
    
        # Getting new action
        s.update()
       
        # apply action and advance environment
        actions = s["Action"]

        les.step(actions,numAgents)
        #les.step(None)
        
        reward = les.getMseReward(numAgents)
        state = les.getState(numAgents)

        s["State"] = state
        s["Reward"] = reward
        
        step += 1
        cumreward += sum(reward)/numAgents


   
    print(cumreward)
    s["Termination"] = "Terminal" if error == 0 else "Truncated"
