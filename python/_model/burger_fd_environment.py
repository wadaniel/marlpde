import os
from Burger import *
from Burger_fd import *
from plotting import makePlot

# reward defaults
episodeCount = 0

# basis defaults
basis = 'hat'

def setup_dns_default(L, N, T, dt, nu , ic, forcing, seed, stepper):
    print(f"Setting up default dns with args ({L}, {N}, {T}, {dt}, {nu}, {ic}, {forcing}, {seed}, {stepper})")
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=T, case=ic, forcing=forcing, noise=0., seed=seed, s=stepper)
    dns.simulate()
    dns.compute_Ek()
    return dns

def environment( s , 
        L,
        T,
        N, 
        gridSize, 
        numActions, 
        dt, 
        nu, 
        episodeLength, 
        ic, 
        spectralReward, 
        forcing, 
        dforce, 
        ssmforce,
        noise, 
        seed, 
        stepper,
        nunoise=False, 
        version=0,
        ssm=False, 
        dsm=False, 
        dns_default = None,
        numAgents = 1):

    global episodeCount
    assert( (ssm and dsm) == False )
 
    #saveEpisode = True if episodeCount >= 1950 else False
    #saveEpisode = True if s["Custom Settings"]["Save Episode"] == "True" else False
    saveEpisode = False
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    #noise = 0. if testing else noise
 
    if testing == True:
        nu = s["Custom Settings"]["Viscosity"]

    ndns = len(dns_default)
    sidx = episodeCount % ndns
    
    if nunoise:
        dns = Burger(L=L, 
                N=N, 
                dt=dt, 
                nu=nu, 
                tend=T, 
                case=ic, 
                forcing=forcing, 
                noise=0., 
                seed=seed+sidx, 
                s=stepper,
                version=version, 
                nunoise=nunoise, 
                numAgents = 1)

        dns.simulate()
        dns.compute_Ek()

        nu = dns.nu
    else:
        dns = dns_default[sidx]
    
    # reward defaults
    rewardFactor = 1. if spectralReward else 1.
     
    #Initialize LES
    sgs = Burger_fd(L=L, 
            N=gridSize, 
            dt=dt, 
            nu=nu, 
            tend=T, 
            case=ic, 
            forcing=forcing, 
            dforce=dforce, 
            ssmforce=ssmforce,
            noise=noise, 
            seed=seed+sidx,
            s=stepper,
            version=version,
            numAgents = numAgents,
            nunoise=nunoise )
   
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
 
    ## get initial state
    state = sgs.getState()
    if(np.isfinite(state).all() == False):
        print("[burger_fd_environment] invalid Initial state")

    s["State"] = state[0] if numAgents == 1 else state

    ## run controlled simulation
    error = 0
    step = 0
    kPrevRelErr = 0.
    nIntermediate = int(T / (dt) / episodeLength)
    assert nIntermediate > 0, "dt or episodeLendth too long"

    cumreward = np.zeros(numAgents)

    while step < episodeLength and error == 0:
    
        # init reward
        reward = 0.
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"] 

        reward = np.zeros(numAgents)

        try:
            for _ in range(nIntermediate):
                sgs.step(actions)
            
                # calculate MSE reward
                if spectralReward == False:
                    reward += rewardFactor*sgs.getMseReward(sgs.offset) / nIntermediate
            
            # get new state
            state = sgs.getState()

        except Exception as e:
            print("[burger_fd_environment] Exception occured during stepping:")
            print(str(e))
            error = 1
            break

        if(np.isfinite(state).all() == False or (np.array(state)>1e6).any()):
            print("[burger_fd_environment] Invalid state detected")
            error = 1
            break

        s["State"] = state[0] if numAgents == 1 else state
    
        # calculate spectral reward
        if spectralReward:
            sgs.compute_Ek()
            kRelErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,1:gridSize//2] - sgs.Ek_ktt[sgs.ioutnum,1:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,1:gridSize//2])**2)
            reward = np.full(numAgents, [rewardFactor*(kPrevRelErr-kRelErr)])
            kPrevRelErr = kRelErr
        
        # accumulate reward
        cumreward += reward

        if (np.isfinite(reward) == False).any():
            print("[burger_fd_environment] Nan reward detected")
            error = 1
            break
    
        else:
            if numAgents > 1:
                s["Reward"] = reward.tolist()
            else:
                s["Reward"] = reward[0]
 
        step += 1

    episodeCount += 1
    print(f"Episode {episodeCount}: {cumreward}")
    #saveEpisode = True if cumreward[0] >= -0.25 else False
    
    if error == 1:
        s["State"] = state[0] if numAgents == 1 else state
        s["Reward"] = -np.inf if numAgents == 1 else [-np.inf]*numAgents
        s["Termination"] = "Truncated"
    
    else:
        s["Termination"] = "Terminal"


    if saveEpisode:
        print("[burger_fd_environment] saving episode..")
        fname = s["Custom Settings"]["Filename"]
        if os.path.isfile(fname):
            print(f"Loading file {fname}")
            npzfile = np.load(fname)
            dns_Ektt = np.vstack((npzfile['dns_Ektt'], dns.Ek_ktt))
            sgs_Ektt = np.vstack((npzfile['sgs_Ektt'], sgs.Ek_ktt))
            sgs_actions = np.vstack((npzfile['sgs_actions'], sgs.actionHistory))
            sgs_u = np.vstack((npzfile['sgs_u'], sgs.uu))
            dns_u = np.vstack((npzfile['dns_u'], dns.uu))
            indeces = np.concatenate((npzfile['indeces'], np.array([sidx])))
        else:
            dns_Ektt = dns.Ek_ktt
            sgs_Ektt = sgs.Ek_ktt
            sgs_actions = sgs.actionHistory
            sgs_u = sgs.uu
            dns_u = dns.uu
            indeces = np.array([sidx])

        print(dns_Ektt.shape)
        print(sgs_Ektt.shape)
        print(sgs_actions.shape)
        print(sgs_u.shape)
        print(dns_u.shape)
        print(indeces)

        np.savez(fname, dns_Ektt=dns_Ektt, sgs_Ektt=sgs_Ektt, sgs_actions=sgs_actions, sgs_u=sgs_u, dns_u=dns_u, indeces=indeces)
        print("[burger_fd_environment] saved!")
        if len(indeces) > 20:
            print("[burger_environment] terminated!")
            sys.exit()

#------------------------------------------------------------------------------
    if testing:
        
        print("[burger_env] Calculating SGS terms from DNS..")
        dns.compute_Sgs(gridSize)

        print("[burger_env] Running URS..")
        base = Burger(L=L, 
                N=gridSize, 
                dt=dt, 
                nu=nu, 
                tend=T, 
                forcing=forcing,
                noise=0.,
                seed=seed,
                dsm=False,
                s=stepper)
        
        ## copy random numbers
        base.randfac1 = dns.randfac1
        base.randfac2 = dns.randfac2
        base.setup_basis(numActions, basis)
        base.setGroundTruth(dns.x, dns.tt, dns.uu)
 
        if spectralReward:
            print("[burger_env] Init spectrum.")
            v0off = dns.v0*np.exp(1j*2*np.pi*sgs.offset*dns.k)
            v0 = np.concatenate((v0off[:((gridSize+1)//2)], v0off[-(gridSize-1)//2:])) * gridSize / dns.N
            base.IC( v0 = v0 )

        else:
            print("[burger_env] Init interpolation.")
            midx = np.argmax(newx)
            if midx == len(newx)-1:
                ic = base.f_truth(newx, 0)
            else:
                ic = np.concatenate(((base.f_truth(newx[:midx+1], 0.)), base.f_truth(newx[midx+1:], 0.)))
            base.IC( u0 = ic )

        # reinit vars
        step = 0
        error = 0
        kPrevRelErr = 0.
        cumreward = np.zeros(numAgents)

        actions = np.zeros(numActions)

        while step < episodeLength and error == 0:
        
            # init reward
            reward = 0.
        
            # apply action and advance environment
            try:
                for _ in range(nIntermediate):
                    base.step(actions)
                 
                    # calculate MSE reward
                    if spectralReward == False:
                        reward += rewardFactor*sgs.getMseReward(sgs.offset) / nIntermediate

            except Exception as e:
                print("[burger_environment] Exception occured during stepping:")
                print(str(e))
                error = 1
                break
            

            # calculate reward
            if spectralReward:
                base.compute_Ek()
                kRelErr = np.mean((np.abs(dns.Ek_ktt[base.ioutnum,:gridSize//2] - base.Ek_ktt[base.ioutnum,:gridSize//2])/dns.Ek_ktt[base.ioutnum,:gridSize//2])**2)
                reward = np.full(numAgents, [rewardFactor*(kPrevRelErr-kRelErr)])
                kPrevRelErr = kRelErr
        

            # accumulate reward
            cumreward += reward

            if (np.isfinite(reward).all() == False):
                print("[burger_environment] Nan reward detected")
                error = 1
                break
     
            step += 1

        print("[burger_environment] uncontrolled cumreward")
        print(cumreward)
        
        makePlot(dns, base, sgs, "burger", spectralReward)
