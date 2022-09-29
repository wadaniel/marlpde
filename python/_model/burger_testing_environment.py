from Burger import *
from plotting import makePlot

# reward defaults
rewardFactor = 1.

# basis defaults
basis = 'hat'

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

    assert( (ssm and dsm) == False )
 
    nu = s["Custom Settings"]["Viscosity"]
    fileName = s["Custom Settings"]["Filename"]

    ndns = len(dns_default)
    nT = int(T//dt)
    dnsSgsTerms = np.zeros((ndns*nT, N))
    sgsTerms = np.zeros((ndns*nT, gridSize))
    relError = np.zeros((ndns*nT, gridSize//2-1))
    cumreward = np.zeros(numAgents)
    print(f"[burger_testing_env] running {ndns} envs..")
    for sidx in range(ndns):
        print(f"{sidx}/{ndns}")
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
         
        print("[burger_resting_env] Calculating SGS terms from DNS..")
        dns.compute_Sgs(gridSize)
        dnsSgsTerms[sidx*nT:(sidx+1)*nT,:] = dns.sgsHistory[:nT, :]
   
        #Initialize LES
        sgs = Burger(L=L, 
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
        nIntermediate = int(T / (dt) / episodeLength)
        assert nIntermediate > 0, "dt or episodeLendth too long"


        while step < episodeLength and error == 0:
    
            # init reward
            reward = 0.
            
            # Getting new action
            s.update()

            # apply action and advance environment
            actions = s["Action"] 

            reward = np.zeros(numAgents)

            for _ in range(nIntermediate):
                sgs.step(actions)
                
            # get new state
            state = sgs.getState()

            s["State"] = state[0] if numAgents == 1 else state
    
            # calculate spectral reward
            sgs.compute_Ek()
            kRelErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,1:gridSize//2] - sgs.Ek_ktt[sgs.ioutnum,1:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,1:gridSize//2])**2)
            reward = np.full(numAgents, [rewardFactor*(kPrevRelErr-kRelErr)])
            kPrevRelErr = kRelErr
        
            # accumulate reward
            cumreward += reward

            s["Reward"] = reward.tolist() if numAgents > 1 else reward[0]
     
            step += 1

        print(cumreward)


        sgsTerms[nT*sidx:nT*(sidx+1),:] = sgs.actionHistory[:nT,:]
        relError[nT*sidx:nT*(sidx+1),:] = (np.abs(dns.Ek_ktt[:nT,1:gridSize//2] - sgs.Ek_ktt[:nT,1:gridSize//2])/dns.Ek_ktt[:nT,1:gridSize//2])**2 
    
    s["Termination"] = "Terminal"

         
#------------------------------------------------------------------------------
    relErrorFile = f'relError_{ic}_{N}_{gridSize}_{version}.npy'
    with open(relErrorFile, 'wb') as f:
        np.save(f, relError)

    sgsTermsFile = f'sgsTerms_{ic}_{N}_{gridSize}_{version}.npy'
    with open(sgsTermsFile, 'wb') as f:
        np.save(f, sgsTerms)
    
    dnsSgsTermsFile = f'dnsSgsTerms_{ic}_{N}_{gridSize}_{version}.npy'
    with open(dnsSgsTermsFile, 'wb') as f:
        np.save(f, dnsSgsTermsFile)
