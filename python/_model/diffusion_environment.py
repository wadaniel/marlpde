from Diffusion import *
from plotting import makeDiffusionPlot

import matplotlib.pyplot as plt 

# defaults
NDNS=512
L = 2*np.pi

# reward defaults
rewardFactor = 1e6

# basis defaults
basis = 'hat'

def setup_dns_default(NDNS, dt, nu, tend, seed):
    print("Setting up default dns with args ({}, {}, {}, {} )".format(NDNS, dt, nu, seed))
    dns = Diffusion(L=L, N=NDNS, dt=dt, nu=nu, tend=tend, case='box', noise=0., implicit = True)
    dns.simulate()
    return dns

def environment( s , N, tEnd, dt_sgs, numActions, nu, episodeLength, ic, dforce, noise, seed, dns_default=None, nunoise=False, tnoise=False, version=0):
    
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    noise = 0.0 if testing else 0.1
        
    if tnoise and testing == False:
        dt_sgs = 0.01+0.04*np.random.uniform()
     
    if testing == True:
        dt_sgs = s["Custom Settings"]["Timestep"]


    #if noise > 0.:
        #dns = Diffusion(L=L, N=NDNS, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, version=version, noise=0., implicit = True)
        #dns.simulate()
    
    #else:
    #    dns = dns_default
 
    # Initialize LES
    les = Diffusion(L=L, N=N, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, version=version, noise=0. )
    les.setup_basis(numActions, basis)

    ## get initial state
    state = les.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    episodeLength = min( int(tEnd/dt_sgs), episodeLength)
    nIntermediate = int(tEnd / dt_sgs / episodeLength)
    
    assert nIntermediate > 0, "Too large timestep"
    cumreward = 0.
    cumMseDiff = 0.

    timestamps = []
    actionHistory = []

    while step < episodeLength and error == 0:
        
        if step > 0:
            # Getting new action
            s.update()

            # apply action and advance environment
            actions = s["Action"]

        else:
            actions = np.zeros(N)

        actionHistory.append(actions)
        timestamps.append(les.t)

        reward = 0.

        try:
            for _ in range(nIntermediate):
                les.step(actions)
         
                # calculate reward
                sol = les.getAnalyticalSolution(les.t)
                uDiffMse = ((sol - les.uu[les.ioutnum,:])**2).mean()
                reward += -rewardFactor*uDiffMse/episodeLength
        
        except Exception as e:
            print("Exception occured:")
            print(str(e))
            error = 1

        # get new state
        newstate = les.getState().flatten().tolist()
        if(np.isfinite(newstate).all() == False):
            print("Nan state detected")
            error = 1
        else:
            state = newstate

        s["State"] = state
    
        # calculate reward
        uDiffMse = ((les.getAnalyticalSolution(les.t) - les.uu[les.ioutnum,:])**2).mean()
        #uDiffMse = ((les.analytical[les.ioutnum,:] - les.uu[les.ioutnum,:])**2).mean()
        cumMseDiff += uDiffMse
        reward = -rewardFactor*uDiffMse
        
        cumreward += reward
        s["Reward"] = reward

        if (np.isfinite(reward) == False):
            print("Nan reward detected")
            error = 1
        
        #if uDiffMse > 1.:
        #    error = 1

        step += 1

    print(step)
    print(dt_sgs)
    print(cumreward)

    if error == 1:
        s["State"] = state
        s["Termination"] = "Truncated"
        s["Reward"] = reward
    
    else:
        s["Termination"] = "Terminal"
    
    if testing:

        fileName = s["Custom Settings"]["Filename"]

        #print("[diffusion_env] Storing sgs to file {}".format(fileName))
        #np.savez(fileName, x = sgs.x, t = sgs.tt, uu = sgs.uu, vv = sgs.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=tEnd, ssm = ssm, dsm = dsm, actions=sgs.actionHistory)
         
#------------------------------------------------------------------------------
        
        #print("[diffusion_env] Calculating SGS terms from DNS..")
    
        # Initialize LES
        base = Diffusion(L=L, N=N, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, version=version, noise=0. )
        base.setup_basis(numActions, basis)
 
        #print("[diffusion_env] Running UGS..")
        #if spectralReward:
        #    print("[burger_env] Init spectrum.")
        #    v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        #    base.IC( v0 = v0 * gridSize / dns.N )

        #else:
        #    print("[burger_env] Init interpolation.")
        #    base.IC( u0 = f_restart(base.x) )

        #base.randfac1 = dns.randfac1
        #base.randfac2 = dns.randfac2
        #base.setup_basis(numActions, basis)
        #base.setGroundTruth(dns.tt, dns.x, dns.uu)
        
        # reinit vars
        error = 0
        step = 0
        prevkMseLogErr = 0.
        kMseLogErr = 0.
        reward = 0.
        cumreward = 0.

        actions = np.zeros(numActions)

        while step < episodeLength and error == 0:

            # apply action and advance environment
            try:
                for _ in range(nIntermediate):
                    base.step(actions)

            except Exception as e:
                print("[diffusion_environment] Exception occured during stepping:")
                print(str(e))
                error = 1
                break
             
            # calculate reward
            uDiffMse = ((base.analytical[base.ioutnum, :] - base.uu[base.ioutnum,:])**2).mean()
            cumMseDiff += uDiffMse
            reward = -rewardFactor*uDiffMse
     
            cumreward += reward

            if (np.isfinite(reward) == False):
                print("[diffusion_environment] Nan reward detected")
                error = 1
                break
     
            step += 1
        
        cumreward /= episodeLength
        print(step)
        print(dt_sgs)
        print("[diffusion_environment] uncontrolled cumreward")
        print(cumreward)
        
        makeDiffusionPlot(base, les, fileName)
