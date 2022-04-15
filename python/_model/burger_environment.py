from Burger import *
from plotting import *

# dns defaults
L    = 2*np.pi
tEnd = 5
basis = 'hat'


def setup_dns_default(N, dt, nu , ic, forcing, seed):
    print("Setting up default dns with args ({}, {}, {}, {}, {}, {})".format(N, dt, nu, ic, forcing, seed))
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, forcing=forcing, noise=0., seed=seed)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()
    return dns

def environment( s , N, gridSize, numActions, dt, nu, episodeLength, ic, spectralReward, forcing, dforce, noise, seed, dns_default = None):
 
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    #noise = 0. if testing else noise
    
    if noise > 0.:
        dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, forcing=forcing, noise=noise, seed=seed)
        dns.simulate()
        dns.fou2real()
        dns.compute_Ek()

    else:
        dns = dns_default
    
    # reward defaults
    rewardFactor = 1. if spectralReward else 1.

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    sgs = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, forcing=forcing, dforce=dforce, noise=0.)
    if spectralReward:
        v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        sgs.IC( v0 = v0 * gridSize / dns.N )
    else:
        sgs.IC( u0 = f_restart(sgs.x) )
 
    sgs.randfac = dns.randfac
    sgs.setup_basis(numActions, basis)
    sgs.setGroundTruth(dns.tt, dns.x, dns.uu)

    ## get initial state
    state = sgs.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tEnd / dt / episodeLength)
    prevkMseLogErr = 0.
    kMseLogErr = 0.
    reward = 0.
    cumreward = 0.

    while step < episodeLength and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
        try:
            for _ in range(nIntermediate):
                sgs.step(actions)

            sgs.compute_Ek()
            sgs.fou2real()
        except Exception as e:
            print("[burger_environment] Exception occured:")
            print(str(e))
            error = 1
            break
        

        # get new state
        newstate = sgs.getState().flatten().tolist()
        if(np.isfinite(newstate).all() == False):
            print("[burger_environment] Nan state detected")
            error = 1
            break
        else:
            state = newstate

        s["State"] = state
    
        # calculate reward
        if spectralReward:
            kMseLogErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,:gridSize//2] - sgs.Ek_ktt[sgs.ioutnum,:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,:gridSize//2])**2)
            reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
            prevkMseLogErr = kMseLogErr

        else:
            uTruthToCoarse = sgs.mapGroundTruth()
            uDiffMse = ((uTruthToCoarse[sgs.ioutnum-nIntermediate:sgs.ioutnum,:] - sgs.uu[sgs.ioutnum-nIntermediate:sgs.ioutnum,:])**2).mean()
            reward = -rewardFactor*uDiffMse

        # accumulat reward
        cumreward += reward

        if (np.isfinite(reward) == False):
            print("[burger_environment] Nan reward detected")
            error = 1
            break
    
        else:
            s["Reward"] = reward
 
        step += 1

    print(cumreward)
    if error == 1:
        s["State"] = state
        s["Termination"] = "Truncated"
        s["Reward"] = -1000 if testing else -np.inf
    
    else:
        s["Termination"] = "Terminal"

    if testing:

        fileName = s["Custom Settings"]["Filename"]
        print("[burger_env] Storing sgs to file {}".format(fileName))
        np.savez(fileName, x = sgs.x, t = sgs.tt, uu = sgs.uu, vv = sgs.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=tEnd, actions=sgs.actionHistory)
         
#------------------------------------------------------------------------------
        
        print("[burger_env] Calculating SGS terms from DNS..")
        dns.compute_Sgs(gridSize)

        print("[burger_env] Running UGS..")
        base = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, forcing=forcing, noise=0.)
        if spectralReward:
            print("[burger_env] Init spectrum.")
            v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
            base.IC( v0 = v0 * gridSize / dns.N )

        else:
            print("[burger_env] Init interpolation.")
            base.IC( u0 = f_restart(base.x) )

        base.randfac = dns.randfac
        base.setup_basis(numActions, basis)
        base.setGroundTruth(dns.tt, dns.x, dns.uu)

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

                base.compute_Ek()
                base.fou2real()
            except Exception as e:
                print("[burger_environment] Exception occured:")
                print(str(e))
                error = 1
                break
            

            # calculate reward
            if spectralReward:
                kMseLogErr = np.mean((np.abs(dns.Ek_ktt[base.ioutnum,:gridSize//2] - base.Ek_ktt[base.ioutnum,:gridSize//2])/dns.Ek_ktt[base.ioutnum,:gridSize//2])**2)
                reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
                prevkMseLogErr = kMseLogErr

            else:
                uTruthToCoarse = base.mapGroundTruth()
                uDiffMse = ((uTruthToCoarse[base.ioutnum-nIntermediate:base.ioutnum,:] - base.uu[base.ioutnum-nIntermediate:base.ioutnum,:])**2).mean()
                reward = -rewardFactor*uDiffMse

            # accumulat reward
            cumreward += reward

            if (np.isfinite(reward) == False):
                print("[burger_environment] Nan reward detected")
                error = 1
                break
     
            step += 1

        print("[burger_environment] uncontrolled cumreward")
        print(cumreward)
        
        makePlot(dns, base, sgs, fileName, spectralReward)
