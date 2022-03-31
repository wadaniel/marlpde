from Burger import *
import matplotlib.pyplot as plt 

# dns defaults
L    = 2*np.pi
tEnd = 5

def setup_dns_default(N, dt, nu , ic, seed):
    print("Setting up default dbs with args ({}, {}, {}, {}, {})".format(N, dt, nu, ic, seed))
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=0., seed=seed)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()
    return dns

# basis defaults
basis = 'hat'

def environment( s , gridSize, numActions, dt, nu, episodeLength, ic, spectralReward, dforce, noise, seed, dns_default = None ):
 
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    noise = 0. if testing else noise   
    
    if noise > 0.:
        dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, dforce=dforce, noise=noise, seed=seed)
        dns.simulate()
        dns.fou2real()
        dns.compute_Ek()
    else:
        dns = dns_default
    
    # reward defaults
    rewardFactor = 0.001 if spectralReward else 1.

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    sgs = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
    if spectralReward:
        v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        sgs.IC( v0 = v0 * gridSize / dns.N )
    else:
        sgs.IC( u0 = f_restart(sgs.x) )

    sgs.setup_basis(numActions, basis)
    sgs.setGroundTruth(dns.tt, dns.x, dns.uu)

    ## get initial state
    state = sgs.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tEnd / dt / episodeLength)
    cumreward = 0.

    timestamps = []
    actionHistory = []

    while step < episodeLength and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
        actionHistory.append(actions)
        timestamps.append(sgs.t)

        try:
            for _ in range(nIntermediate):
                sgs.step(actions)

            sgs.compute_Ek()
            sgs.fou2real()
        except Exception as e:
            print("Exception occured:")
            print(str(e))
            error = 1
            break
        

        # get new state
        newstate = sgs.getState().flatten().tolist()
        if(np.isfinite(newstate).all() == False):
            print("Nan state detected")
            error = 1
            break
        else:
            state = newstate

        s["State"] = state
    
        # calculate reward
        if spectralReward:
            kMseLogErr = np.mean((np.log(dns.Ek_kt[sgs.ioutnum,:gridSize]) - np.log(sgs.Ek_kt[sgs.ioutnum,:gridSize]))**2)
            reward = -rewardFactor*kMseLogErr
        else:
            reward = rewardFactor*sgs.getMseReward()

        # accumulat reward
        cumreward += reward

        if (np.isfinite(reward) == False):
            print("Nan reward detected")
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
        actionHistory = np.array(actionHistory)
        print("Storing sgs to file {}".format(fileName))
        np.savez(fileName, x = sgs.x, t = sgs.tt, uu = sgs.uu, vv = sgs.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=tEnd, actions=actionHistory)
         
        print("Running uncontrolled SGS..")
        base = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
        if spectralReward:
            print("Init spectrum.")
            v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
            base.IC( v0 = v0 * gridSize / dns.N )

        else:
            print("Init interpolation.")
            base.IC( u0 = f_restart(base.x) )


        base.simulate()
        base.fou2real()
        base.compute_Ek()
       
        N  = dns.N
        nt = int(tEnd/dt)
        k1 = dns.k[:N//2]
        k2 = sgs.k[:gridSize//2]

        colors = plt.cm.jet(np.linspace(0,1,5))

        time = np.arange(tEnd/dt+1)*dt

        fig1, axs1 = plt.subplots(3, 6, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(15,15))

        umax = max(dns.uu.max(), base.uu.max(), sgs.uu.max())
        umin = min(dns.uu.min(), base.uu.min(), sgs.uu.min())
        ulevels = np.linspace(umin, umax, 50)

#------------------------------------------------------------------------------
        print("plot DNS")

        idx = 0
        axs1[0,0].contourf(dns.x, dns.tt, dns.uu, ulevels)

        axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[0,0:N//2]),':', color=colors[idx])
        axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[nt//2,0:N//2]),'--', color=colors[idx])
        axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[-1,0:N//2]),'-', color=colors[idx])
        axs1[0,3].plot(k1[2:-10], 1e-5*k1[2:-10]**(-2),'--', linewidth=0.5)
        axs1[0,3].set_xscale('log')
        axs1[0,3].set_yscale('log')

#------------------------------------------------------------------------------

        errBaseEk_t = dns.Ek_t - base.Ek_t
        errBaseEk_tt = dns.Ek_tt - base.Ek_tt

        f_dns = interpolate.interp2d(dns.x, dns.tt, dns.uu, kind='cubic')
        udns_int = f_dns(base.x, base.tt)
        errBaseU = np.abs(base.uu-udns_int)
        mseBaseU_t = np.mean(errBaseU**2, axis=1)
        
        errEk_t = dns.Ek_t - sgs.Ek_t
        errEk_tt = dns.Ek_tt - sgs.Ek_tt
        
        errU = np.abs(sgs.uu-udns_int)
        mseU_t = np.mean(errU**2, axis=1)
 
#------------------------------------------------------------------------------

        emax = max(errBaseU.max(), errU.max())
        emin = min(errBaseU.min(), errU.min())
        elevels = np.linspace(emin, emax, 50)
        
#------------------------------------------------------------------------------
        print("plot baseline")
        idx = idx + 1
 
        # Plot solution
        axs1[idx,0].contourf(base.x, base.tt, base.uu, ulevels)
  
        # Plot difference to dns
        axs1[idx,1].contourf(base.x, base.tt, errBaseU, elevels)

        # Plot instanteneous energy and time averaged energy
        axs1[idx,2].plot(base.tt, mseBaseU_t, 'r-')
        axs1[idx,2].set_yscale('log')
        axs1[idx,2].set_ylim([1e-8,None])

        # Plot energy spectrum at start, mid and end of simulation
        axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[0,0:gridSize//2]),':',color=colors[idx])
        axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[nt//2,0:gridSize//2]),'--',color=colors[idx])
        axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[-1,0:gridSize//2]),'-',color=colors[idx])
        axs1[idx,3].set_xscale('log')
        axs1[idx,3].set_yscale('log')
        axs1[idx,3].set_ylim([1e-8,None])
 
        # Plot energy spectrum difference
        axs1[idx,4].plot(k2[1:], np.abs(dns.Ek_ktt[0,1:gridSize//2] - base.Ek_ktt[0,1:gridSize//2]),'r:')
        axs1[idx,4].plot(k2, np.abs(dns.Ek_ktt[nt//2,0:gridSize//2] - base.Ek_ktt[gridSize//2,0:gridSize//2]),'r--')
        axs1[idx,4].plot(k2, np.abs(dns.Ek_ktt[-1,0:gridSize//2] - base.Ek_ktt[-1,0:gridSize//2]),'r')
        axs1[idx,4].set_xscale('log')
        axs1[idx,4].set_yscale('log')
        axs1[idx,4].set_ylim([1e-14,None])

#------------------------------------------------------------------------------

        print("plot sgs")
        idx = idx + 1
 
        # Plot solution
        axs1[idx,0].contourf(sgs.x, sgs.tt, sgs.uu, ulevels)
  
        # Plot difference to dns
        axs1[idx,1].contourf(sgs.x, sgs.tt, errU, elevels)

        # Plot instanteneous energy and time averaged energy
        axs1[idx,2].plot(sgs.tt, mseU_t, 'r-')
        axs1[idx,2].set_yscale('log')
        axs1[idx,2].set_ylim([1e-8,None])

        # Plot time averaged energy spectrum at start, mid and end of simulation
        axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[0,0:gridSize//2]),':',color=colors[idx])
        axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[nt//2,0:gridSize//2]),'--',color=colors[idx])
        axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[-1,0:gridSize//2]),'-',color=colors[idx])
        axs1[idx,3].set_xscale('log')
        axs1[idx,3].set_yscale('log')
        axs1[idx,3].set_ylim([1e-8,None])
 
        # Plot time averaged energy spectrum difference
        axs1[idx,4].plot(k2[1:], np.abs(dns.Ek_ktt[0,1:gridSize//2] - sgs.Ek_ktt[0,1:gridSize//2]),'r:')
        axs1[idx,4].plot(k2, np.abs(dns.Ek_ktt[nt//2,0:gridSize//2] - sgs.Ek_ktt[gridSize//2,0:gridSize//2]),'r--')
        axs1[idx,4].plot(k2, np.abs(dns.Ek_ktt[-1,0:gridSize//2] - sgs.Ek_ktt[-1,0:gridSize//2]),'r')
        axs1[idx,4].set_xscale('log')
        axs1[idx,4].set_yscale('log')
        axs1[idx,4].set_ylim([1e-14,None])

        acolors = plt.cm.coolwarm(np.linspace(0,1,numActions))
        for i in range(numActions):
            axs1[idx,5].plot(timestamps, actionHistory[:,i], color=acolors[i])

        plt.tight_layout()
        figName = fileName + ".png"
        fig1.savefig(figName)

#------------------------------------------------------------------------------

        figName2 = fileName + "_evolution.pdf"
        print("Plotting {} ...".format(figName2))
        
        fig2, axs2 = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
        for i in range(16):
            t = i * tEnd / 16
            tidx = int(t/dt)
            k = int(i / 4)
            l = i % 4
            
            axs2[k,l].plot(base.x, base.uu[tidx,:], '-b')
            axs2[k,l].plot(sgs.x, sgs.uu[tidx,:], '-r')
            axs2[k,l].plot(dns.x, dns.uu[tidx,:], '--k')

        fig2.savefig(figName2)
