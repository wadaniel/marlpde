from Burger import *
import matplotlib.pyplot as plt 

# dns defaults
N    = 512
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.02

# reward structure
spectralReward = False
spectralLogReward = False

# reward defaults
rewardFactor = 0.001 if spectralReward else 1.
rewardFactor = 0.001 if spectralLogReward else rewardFactor

dns_default = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case='turbulence', noise=0., seed=42)
dns_default.simulate()
dns_default.fou2real()
dns_default.compute_Ek()

# basis defaults
basis = 'hat'

def environment( s , gridSize, numActions, episodeLength, ic, dforce, noise, seed ):
 
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    noise = 0. if testing else noise   
    
    if noise > 0.:
        dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, dforce=dforce, noise=noise, seed=seed)
        dns.simulate()
        dns.fou2real()
        dns.compute_Ek()
    else:
        dns = dns_default

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
    if spectralReward or spectralLogReward:
        v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        les.IC( v0 = v0 * gridSize / N )
    else:
        les.IC( u0 = f_restart(les.x) )

    les.setup_basis(numActions, basis)
    les.setGroundTruth(dns.tt, dns.x, dns.uu)

    ## get initial state
    state = les.getState().flatten().tolist()
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
        timestamps.append(les.t)

        try:
            for _ in range(nIntermediate):
                les.step(actions)

            les.compute_Ek()
            les.fou2real()
        except Exception as e:
            print("Exception occured:")
            print(str(e))
            error = 1
            break
        

        # get new state
        newstate = les.getState().flatten().tolist()
        if(np.isfinite(newstate).all() == False):
            print("Nan state detected")
            error = 1
            break
        else:
            state = newstate

        s["State"] = state
    
        # calculate reward
        if spectralReward:
            # Time-averaged energy spectrum as a function of wavenumber
            kMseErr = np.mean((dns.Ek_kt[les.ioutnum,:gridSize] - les.Ek_kt[les.ioutnum,:gridSize])**2)
            reward = -rewardFactor*kMseErr
    
        elif spectralLogReward:
            kMseLogErr = np.mean((np.log(dns.Ek_kt[les.ioutnum,:gridSize]) - np.log(les.Ek_kt[les.ioutnum,:gridSize]))**2)
            reward = -rewardFactor*kMseLogErr

        else:
            reward = rewardFactor*les.getMseReward()

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
        print("Storing les to file {}".format(fileName))
        np.savez(fileName, x = les.x, t = les.tt, uu = les.uu, vv = les.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=tEnd, actions=actionHistory)
         
        print("Running uncontrolled SGS..")
        base = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
        if spectralReward or spectralLogReward:
            print("Init spectrum.")
            v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
            base.IC( v0 = v0 * gridSize / N )

        else:
            print("Init interpolation.")
            base.IC( u0 = f_restart(base.x) )


        base.simulate()
        base.fou2real()
        base.compute_Ek()
       
        k1 = dns.k[:N//2]
        k2 = les.k[:gridSize//2]

        time = np.arange(tEnd/dt+1)*dt

        fig1, axs1 = plt.subplots(3, 7, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(15,15))

        umax = max(dns.uu.max(), base.uu.max(), les.uu.max())
        umin = min(dns.uu.min(), base.uu.min(), les.uu.min())
        ulevels = np.linspace(umin, umax, 50)

#------------------------------------------------------------------------------
        print("plot DNS")

        axs1[0,0].contourf(dns.x, dns.tt, dns.uu, ulevels)

        axs1[0,2].plot(time, dns.Ek_t)
        axs1[0,2].plot(time, dns.Ek_tt)

        axs1[0,4].plot(k1, np.abs(dns.Ek_ktt[0,0:N//2]),':b')
        axs1[0,4].plot(k1, np.abs(dns.Ek_ktt[dns.ioutnum//2,0:N//2]),'--b')
        axs1[0,4].plot(k1, np.abs(dns.Ek_ktt[-1,0:N//2]),'b')
        axs1[0,4].set_xscale('log')
        axs1[0,4].set_yscale('log')
 
#------------------------------------------------------------------------------

        errBaseEk_t = dns.Ek_t - base.Ek_t
        errBaseEk_tt = dns.Ek_tt - base.Ek_tt

        f_dns = interpolate.interp2d(dns.x, dns.tt, dns.uu, kind='cubic')
        udns_int = f_dns(base.x, base.tt)
        errBaseU = np.abs(base.uu-udns_int)
        mseBaseU_t = np.mean(errBaseU**2, axis=1)

#------------------------------------------------------------------------------

        errEk_t = dns.Ek_t - les.Ek_t
        errEk_tt = dns.Ek_tt - les.Ek_tt
        errU = np.abs(les.uu-udns_int)
        mseU_t = np.mean(errU**2, axis=1)
 
        emax = max(errBaseU.max(), errU.max())
        emin = min(errBaseU.min(), errU.min())
        elevels = np.linspace(emin, emax, 50)
        
#------------------------------------------------------------------------------
        print("plot baseline")
        
 
        idx = 1
        # Plot solution
        axs1[idx,0].contourf(base.x, base.tt, base.uu, ulevels)
  
        # Plot difference to dns
        axs1[idx,1].contourf(base.x, base.tt, errBaseU, elevels)

        # Plot instanteneous energy and time averaged energy
        axs1[idx,2].plot(time, base.Ek_t)
        axs1[idx,2].plot(time, base.Ek_tt)
     
        # Plot energy and field differences
        if spectralReward or spectralLogReward:
            axs1[idx,3].plot(time, np.cumsum(np.mean((np.log(dns.Ek_kt[:,0:gridSize//2]) - np.log(base.Ek_kt[:,0:gridSize//2]))**2,axis=1))/np.arange(1,len(time)+1),'-r')
        else:
            axs1[idx,3].plot(time, mseBaseU_t)
            axs1[idx,3].set_yscale('log')

        # Plot energy spectrum at start, mid and end of simulation
        axs1[idx,4].plot(k2, np.abs(base.Ek_ktt[0,0:gridSize//2]),':b')
        axs1[idx,4].plot(k2, np.abs(base.Ek_ktt[dns.ioutnum//2,0:gridSize//2]),'--b')
        axs1[idx,4].plot(k2, np.abs(base.Ek_ktt[-1,0:gridSize//2]),'-b')
 
        # Plot energy spectrum difference
        axs1[idx,5].plot(k2, (np.log(dns.Ek_ktt[0,0:gridSize//2]) - np.log(base.Ek_ktt[0,0:gridSize//2]))**2,':r')
        axs1[idx,5].plot(k2, (np.log(dns.Ek_ktt[dns.ioutnum//2,0:gridSize//2]) - np.log(base.Ek_ktt[dns.ioutnum//2,0:gridSize//2]))**2,'--r')
        axs1[idx,5].plot(k2, (np.log(dns.Ek_ktt[-1,0:gridSize//2]) - np.log(base.Ek_ktt[-1,0:gridSize//2]))**2,'-r')
        axs1[idx,5].set_xscale('log')
        #axs1[idx,5].set_yscale('log')
 
#------------------------------------------------------------------------------
        print("plot les")
        
        idx += 1
        # Plot solution
        axs1[idx,0].contourf(les.x, les.tt, les.uu, ulevels)
        
        # Plot difference to dns
        axs1[idx,1].contourf(les.x, les.tt, errU, elevels)
 

        # Plot instanteneous energy and time averaged energy
        axs1[idx,2].plot(time, les.Ek_t)
        axs1[idx,2].plot(time, les.Ek_tt)
     
        # Plot energy differences
        if spectralReward or spectralLogReward:
            axs1[idx,3].plot(time, np.cumsum(np.mean((np.log(dns.Ek_kt[:,0:gridSize//2]) - np.log(les.Ek_kt[:,0:gridSize//2]))**2,axis=1))/np.arange(1,len(time)+1),'-r')
        else:
            axs1[idx,3].plot(time, mseU_t)

        # Plot energy spectrum at start, mid and end of simulation
        axs1[idx,4].plot(k2, np.abs(les.Ek_ktt[0,0:gridSize//2]),':b')
        axs1[idx,4].plot(k2, np.abs(les.Ek_ktt[dns.ioutnum//2,0:gridSize//2]),'--b')
        axs1[idx,4].plot(k2, np.abs(les.Ek_ktt[-1,0:gridSize//2]),'-b')
  
        # Plot energy spectrum difference
        #axs1[idx,5].plot(k2, np.abs(dns.Ek_ktt[0,0:gridSize//2] - les.Ek_ktt[0,0:gridSize//2]),':r')
        #axs1[idx,5].plot(k2, np.abs(dns.Ek_ktt[dns.ioutnum//2,0:gridSize//2] - les.Ek_ktt[dns.ioutnum//2,0:gridSize//2]),'--r')
        #axs1[idx,5].plot(k2, np.abs(dns.Ek_ktt[-1,0:gridSize//2] - les.Ek_ktt[-1,0:gridSize//2]),'-r')
        axs1[idx,5].plot(k2, (np.log(dns.Ek_ktt[0,0:gridSize//2]) - np.log(les.Ek_ktt[0,0:gridSize//2]))**2,':r')
        axs1[idx,5].plot(k2, (np.log(dns.Ek_ktt[dns.ioutnum//2,0:gridSize//2]) - np.log(les.Ek_ktt[dns.ioutnum//2,0:gridSize//2]))**2,'--r')
        axs1[idx,5].plot(k2, (np.log(dns.Ek_ktt[-1,0:gridSize//2]) - np.log(les.Ek_ktt[-1,0:gridSize//2]))**2,'-r')
        
        colors = plt.cm.coolwarm(np.linspace(0,1,numActions))
        for i in range(numActions):
            axs1[idx,6].plot(timestamps, actionHistory[:,i], color=colors[i])



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
            axs2[k,l].plot(les.x, les.uu[tidx,:], '-r')
            axs2[k,l].plot(dns.x, dns.uu[tidx,:], '--k')

        fig2.savefig(figName2)
