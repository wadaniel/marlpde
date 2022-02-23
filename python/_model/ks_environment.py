from KS import *
import matplotlib.pyplot as plt

N    = 1024
L    = 22/(2*np.pi)
nu   = 1.0
dt   = 0.01
tTransient = 50
tEnd = 100
tSim = tEnd - tTransient
nSimSteps = int(tSim/dt)

rewardFactor = 10

# DNS baseline
dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient)

def environment( s , gridSize, numActions, episodeLength ):
    
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
 
    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')
 
    # Initialize LES
    les = KS(L=L, N = gridSize, dt=dt, nu=1.0, tend=tSim)
    les.IC( u0 = f_restart(les.x))
    les.setup_basis(numActions)

    ## get initial state
    state = les.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tSim / dt / episodeLength)
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
        except Exception as e:
            print("Exception occured:")
            print(str(e))
            error = 1
            break
        
        # get new state
        state = les.getState().flatten().tolist()
        if(np.isnan(state).any() == True):
            print("Nan state detected")
            error = 1
            break
        s["State"] = state

        # calculate reward from energy
        reward = -rewardFactor*(np.abs(les.Ek_tt[step*nIntermediate]-dns.Ek_tt[step*nIntermediate]))
        cumreward += reward
        
        if (np.isnan(reward)):
            print("Nan reward detected")
            error = 1
            break

        s["Reward"] = reward
        step += 1

    print(cumreward)
    if error == 1:
        s["Termination"] = "Truncated"
        s["Reward"] = -500
    
    else:
        s["Termination"] = "Terminal"
    
    if testing:

        # store controlled LES
        fileName = s["Custom Settings"]["Filename"]
        print("Storing les to file {}".format(fileName))
        np.savez(fileName, x = les.x, t = les.tt, uu = les.uu, vv = les.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=tEnd)
         
        print("Running uncontrolled SGS..")
        base = KS(L=L, N = gridSize, dt=dt, nu=1.0, tend=tEnd-tTransient)
        base.IC( u0 = f_restart(les.x))
        base.simulate()
        base.fou2real()
        base.compute_Ek()
  
        # interpolation DNS
        f_dns = interpolate.interp2d(dns.x, dns.tt, dns.uu, kind='cubic')
        udns_int = f_dns(base.x, base.tt)
      
        print("plot DNS")
        k1 = dns.k[:N//2]

        fig, axs = plt.subplots(3, 6, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(15,15))
        c1 = axs[0,0].contourf(dns.x, dns.tt, dns.uu, 50)

        axs[0,2].plot(dns.tt, dns.Ek_t)
        axs[0,2].plot(dns.tt, dns.Ek_tt)

        axs[0,4].plot(k1, np.abs(dns.Ek_ktt[0,0:N//2]),'b:')
        axs[0,4].plot(k1, np.abs(dns.Ek_ktt[nSimSteps//2,0:N//2]),'b--')
        axs[0,4].plot(k1, np.abs(dns.Ek_ktt[-1,0:N//2]),'b')
        axs[0,4].set_xscale('log')
        axs[0,4].set_yscale('log')
 
#------------------------------------------------------------------------------

        # caclulate error of baseline
        errBaseEk_t = dns.Ek_t - base.Ek_t
        errBaseEk_tt = dns.Ek_tt - base.Ek_tt
        errBaseU = np.abs(base.uu-udns_int)

#------------------------------------------------------------------------------

        # caclulate error of contolled les
        errEk_t = dns.Ek_t - les.Ek_t
        errEk_tt = dns.Ek_tt - les.Ek_tt
        errU = np.abs(les.uu-udns_int)
 
#------------------------------------------------------------------------------

        k2 = les.k[:gridSize//2]
 
        idx = 1
        # Plot solution
        axs[idx,0].contourf(base.x, base.tt, base.uu, c1.levels)
        
        # Plot difference to dns
        c2 = axs[idx,1].contourf(les.x, base.tt, errBaseU, 50)

        # Plot instanteneous energy and time averaged energy
        axs[idx,2].plot(base.tt, base.Ek_t)
        axs[idx,2].plot(base.tt, base.Ek_tt)
     
        # Plot energy differences
        axs[idx,3].plot(base.tt, errBaseEk_t)
        axs[idx,3].plot(base.tt, errBaseEk_tt)

        # Plot energy spectrum at start, mid and end of simulation
        axs[idx,4].plot(k2, np.abs(base.Ek_ktt[0,0:gridSize//2]),'b:')
        axs[idx,4].plot(k2, np.abs(base.Ek_ktt[nSimSteps//2,0:gridSize//2]),'b--')
        axs[idx,4].plot(k2, np.abs(base.Ek_ktt[-1,0:gridSize//2]),'b')
        axs[idx,4].set_xscale('log')
        axs[idx,4].set_yscale('log')

#------------------------------------------------------------------------------
       
        idx += 1
        # Plot solution
        axs[idx,0].contourf(les.x, les.tt, les.uu, c1.levels)
        
        # Plot difference to dns
        axs[idx,1].contourf(les.x, les.tt, errU, c2.levels)

        # Plot instanteneous energy and time averaged energy
        axs[idx,2].plot(les.tt, les.Ek_t)
        axs[idx,2].plot(les.tt, les.Ek_tt)
     
        # Plot energy differences
        axs[idx,3].plot(les.tt, errEk_t)
        axs[idx,3].plot(les.tt, errEk_tt)
        axs[idx,3].set_yscale('log')

        # Plot energy spectrum at start, mid and end of simulation
        axs[idx,4].plot(k2, np.abs(les.Ek_ktt[0,0:gridSize//2]),'b:')
        axs[idx,4].plot(k2, np.abs(les.Ek_ktt[nSimSteps//2,0:gridSize//2]),'b--')
        axs[idx,4].plot(k2, np.abs(les.Ek_ktt[-1,0:gridSize//2]),'b')
        axs[idx,4].set_xscale('log')
        axs[idx,4].set_yscale('log')
 
        # Plot energy spectrum at start, mid and end of simulation
        axs[idx,5].plot(timestamps, actionHistory,'g')
        
        figName = fileName + '.png'
        fig.savefig(figName)
