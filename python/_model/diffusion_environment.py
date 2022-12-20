from Diffusion import *
from plotting import makeDiffusionPlot

import matplotlib.pyplot as plt 

# defaults
NDNS=512
L = 2*np.pi

# reward defaults
rewardFactor = 1e2

# basis defaults
basis = 'hat'

def setup_dns_default(NDNS, dt, nu, tend, seed):
    print("[diffusion_environment] Setting up default dns with args ({}, {}, {}, {} )".format(NDNS, dt, nu, seed))
    dns = Diffusion(L=L, N=NDNS, dt=dt, nu=nu, tend=tend, case='sinus', noise=0., implicit = True)
    dns.simulate()
    return dns

def environment( s , N, tEnd, dtSgs, nu, episodeLength, ic, noise, seed, dnsDefault, nunoise=False, version=0):
    
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False

    les = Diffusion(L=L, N=N, dt=dtSgs, nu=nu, tend=tEnd, case=ic, noise=0., seed=seed)
    les.setGroundTruth(dnsDefault.tt, dnsDefault.x, dnsDefault.uu)

    step = 0
    error = 0
    nIntermediate = int(tEnd / dtSgs / episodeLength)
    assert nIntermediate > 0
    cumreward = 0.


    state = les.getState()
    s["State"] = state

    while step < episodeLength and error == 0:
    
        # Getting new action
        s.update()
       
        # apply action and advance environment
        actions = s["Action"]
        actions = np.array(actions)

        # reweighting
        actions = actions - sum(actions)
        
        try:
            for _ in range(nIntermediate):
                les.step(actions)
        except Exception as e:
            print("Exception occured:")
            print(str(e))
            error = 1
            break
        
        res = les.mapGroundTruth()
        reward = np.mean((res[-1,:] - les.uu[les.ioutnum,:])**2)
        
        state = les.getState()

        s["State"] = state
        s["Reward"] = reward*rewardFactor
        
        if (np.isnan(reward)):
            print("[diffusion_environment] Nan reward detected")
            error = 1
            break
        
        step += 1
        cumreward += reward


   
    print(cumreward)
    s["Termination"] = "Terminal" if error == 0 else "Truncated"
