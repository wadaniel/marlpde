from Advection import *
import plotting_diffusion as dplt

import matplotlib.pyplot as plt 

# defaults
NDNS=512
L = 2*np.pi

# basis defaults
basis = 'hat'

def setup_dns_default(ic, NDNS, dt, nu, tend, seed):
    print("[diffusion_environment] Setting up default dns with args ({}, {}, {}, {} )".format(NDNS, dt, nu, seed))
    dns = Advection(L=L, N=NDNS, dt=dt, nu=nu, tend=tend, case=ic, noise=0., implicit = True)
    dns.simulate()
    return dns

def environment( s , N, tEnd, dtSgs, nu, episodeLength, ic, noise, seed, dnsDefault, nunoise=False, numAgents=1, version=0):
    
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False

    les = Advection(L=L, N=N, dt=dtSgs, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
    #les.setGroundTruth(dnsDefault.tt, dnsDefault.x, dnsDefault.uu)

    step = 0
    stop = False
    cumreward = 0.

    # allowing 10-20 steps
    bonus = { 128: 1e-2,
              64: 5e-2,
              32: 5e-2, 
              16: 1e-1,
              8: 1e-1} 

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


   
    print(f"steps: {step}, cumreward {cumreward}")
    s["Termination"] = "Terminal" 
    
    if testing:

        dplt.plotEvolution(les)
        dplt.plotError(les)
        dplt.plotActionField(les)
        dplt.plotActionDistribution(les)
        dplt.plotDiffusionField(les)



 
