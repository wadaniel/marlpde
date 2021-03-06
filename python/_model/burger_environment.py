from Burger import *
from plotting import makePlot

# reward defaults
rewardFactor = 1.

# basis defaults
basis = 'hat'

def setup_dns_default(L, N, T, dt, nu , ic, forcing, seed, stepper):
    print(f"Setting up default dns with args ({L}, {N}, {T}, {dt}, {nu}, {ic} {forcing}, {seed}, {stepper})")
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
        noise, 
        seed, 
        stepper,
        nunoise=False, 
        version=0,
        ssm=False, 
        dsm=False, 
        dns_default = None,
        numAgents = 1):

    assert( (ssm and dsm) == False )
 
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    noise = 0. if testing else noise
 
    if testing == True:
        nu = s["Custom Settings"]["Viscosity"]

    if nunoise:
        dns = Burger(L=L, 
                N=N, 
                dt=dt, 
                nu=nu, 
                tend=T, 
                case=ic, 
                forcing=forcing, 
                noise=0., 
                seed=seed, 
                s=stepper,
                version=version, 
                nunoise=nunoise, 
                numAgents = 1)

        dns.simulate()
        dns.compute_Ek()

        nu = dns.nu
    else:
        dns = dns_default
    
    # reward defaults
    rewardFactor = 1. if spectralReward else 1.
     
    #Initialize LES
    sgs = Burger(L=L, 
            N=gridSize, 
            dt=dt*stepper, 
            nu=nu, 
            tend=T, 
            case=ic, 
            forcing=forcing, 
            dforce=dforce, 
            noise=noise, 
            seed=seed,
            s=stepper,
            version=version,
            numAgents = numAgents)
   
    ## copy random numbers
    sgs.randfac1 = dns.randfac1
    sgs.randfac2 = dns.randfac2

    sgs.setup_basis(numActions, basis)
    sgs.setGroundTruth(dns.x, dns.tt, dns.uu)
 
    newx = sgs.x + sgs.offset
    newx[newx>L] = newx[newx>L] - L
    newx[newx<0] = newx[newx<0] + L

    if spectralReward:
        v0off = dns.v0*np.exp(1j*2*np.pi*sgs.offset*dns.k)
        v0 = np.concatenate((v0off[:((gridSize+1)//2)], v0off[-(gridSize-1)//2:])) * gridSize / dns.N
        sgs.IC( v0 = v0 )
    else:
        midx = np.argmax(newx)
        if midx == len(newx)-1:
            ic = sgs.f_truth(newx, 0)
        else:
            ic = np.concatenate(((sgs.f_truth(newx[:midx+1], 0.)), sgs.f_truth(newx[midx+1:], 0.)))
        sgs.IC( u0 = ic )
 
    ## get initial state
    state = sgs.getState()
    s["State"] = state[0] if numAgents == 1 else state

    ## run controlled simulation
    error = 0
    step = 0
    kPrevRelErr = 0.
    nIntermediate = int(T / (dt*stepper) / episodeLength)
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
        
        except Exception as e:
            print("[burger_environment] Exception occured during stepping:")
            print(str(e))
            error = 1
            break
        

        # get new state
        state = sgs.getState()
        if(np.isfinite(state).all() == False):
            print("[burger_environment] Nan state detected")
            error = 1
            break

        s["State"] = state[0] if numAgents == 1 else state
    
        # calculate spectral reward
        if spectralReward:
            sgs.compute_Ek()
            kRelErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,1:gridSize//2] - sgs.Ek_ktt[sgs.ioutnum,1:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,1:gridSize//2])**2)
            #print(sgs.Ek_ktt[sgs.ioutnum,:gridSize//2])
            #print(dns.Ek_ktt[sgs.ioutnum,:gridSize//2])
            reward = np.full(numAgents, [rewardFactor*(kPrevRelErr-kRelErr)])
            kPrevRelErr = kRelErr
        
        # accumulate reward
        cumreward += reward

        if (np.isfinite(reward) == False).any():
            print("[burger_environment] Nan reward detected")
            error = 1
            break
    
        else:
            if numAgents > 1:
                s["Reward"] = reward.tolist()
            else:
                s["Reward"] = reward[0]
 
        step += 1

    print(cumreward)
    if error == 1:
        s["State"] = state[0] if numAgents == 1 else state
        s["Reward"] = -np.inf if numAgents == 1 else [-np.inf]*numAgents
        s["Termination"] = "Truncated"
    
    else:
        s["Termination"] = "Terminal"

    if testing:
        
        fileName = s["Custom Settings"]["Filename"]

        #print("[burger_env] Storing sgs to file {}".format(fileName))
        #np.savez(fileName, x = sgs.x, t = sgs.tt, uu = sgs.uu, vv = sgs.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=T, ssm = ssm, dsm = dsm, actions=sgs.actionHistory)
         
#------------------------------------------------------------------------------
        
        print("[burger_env] Calculating SGS terms from DNS..")
        dns.compute_Sgs(gridSize)

        print("[burger_env] Running URS..")
        base = Burger(L=L, 
                N=gridSize, 
                dt=dt*stepper, 
                nu=nu, 
                tend=T, 
                forcing=forcing,
                noise=0.,
                seed=seed,
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
                #print(dns.Ek_ktt[base.ioutnum,:gridSize//2])
                #print(dns.Ek_ktt[base.ioutnum,:gridSize//2])
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
        
        makePlot(dns, base, sgs, fileName, spectralReward)
