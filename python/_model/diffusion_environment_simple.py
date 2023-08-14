from Diffusion import *
import plotting_diffusion as dplt

import matplotlib.pyplot as plt 

# defaults
NDNS=512
L = 2*np.pi

# basis defaults
basis = 'hat'

def setup_dns_default(ic, NDNS, dt, nu, tend, seed):
    print("[diffusion_environment_simple] Setting up default dns with args ({}, {}, {}, {} )".format(NDNS, dt, nu, seed))
    dns = Diffusion(L=L, N=NDNS, dt=dt, nu=nu, tend=tend, case=ic, noise=0., implicit = True)
    dns.simulate()
    print("[diffusion_environment_simple] Done!")
    return dns

def environment( s , N, tEnd, dtSgs, nu, episodeLength, ic, noise, seed, dnsDefault, nunoise=False, numAgents=1, version=0):

    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False

    les = Diffusion(L=L, N=N, dt=dtSgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
    les.setGroundTruth(dnsDefault.tt, dnsDefault.x, dnsDefault.uu)

    step = 0
    stop = False
    cumreward = 0.

    # allowing initial 10-20 steps
    bonus = {
              128: 5e-4,
              64: 5e-5,
              32: 5e-5, 
              16: 5e-5,
              8: 5e-5,
              4: 5e-5,
              2: 5e-5,
              1: 5e-5} 

    state = les.getState(numAgents)
    s["State"] = state

    while step < episodeLength and stop == False:
    
        # Getting new action
        s.update()
       
        # apply action and advance environment
        actions = s["Action"]

        les.step(actions,numAgents)
        #les.step(None)
        
        reward = les.getMseReward(numAgents)

        # grant 'survival bonus'
        if numAgents > 1:
            reward = [r + bonus[N] for r in reward]
        else:
            reward += bonus[N]
        state = les.getState(numAgents)

        s["State"] = state
        s["Reward"] = reward
        
        step += 1
        cumreward += reward if numAgents == 1 else sum(reward)/numAgents
        if cumreward < 0.:
            stop = True

    print(f"[diffusion_environment_simple] steps: {step}, cumreward {cumreward}")
    s["Termination"] = "Terminal" 
    
    if testing:

        dplt.plotEvolution(les)
        dplt.plotActionField(les)
        dplt.plotActionDistribution(les)
        dplt.plotDiffusionField(les)



 
