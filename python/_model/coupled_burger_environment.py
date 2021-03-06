from Burger import *
import matplotlib.pyplot as plt 

from scipy.fftpack import fft, ifft

# dns defaults
N    = 512
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.01

# reward defaults
rewardFactor = 1.
basereward = True

# basis defaults
basis = 'hat'

def environment( s , gridSize, numActions, episodeLength, ic, noise, seed ):
 
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    noise = 0. if testing else noise
    
    seed = s["Sample Id"]
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')
    base0 = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
 
    if basereward == True:
        try:
            base0.IC( u0 = f_restart(base0.x) )
            base0.simulate()
            base0.fou2real()
     
        except Exception as e:
            print("Exception occured in base0:")
            print(str(e))
            error = 1

    # Initialize LES
    les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
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
    rewardHistory = []

    while step < episodeLength and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
        actionHistory.append(actions)
        timestamps.append(les.t)
                
        vBase = les.vv[les.ioutnum,:]
        uBase = les.uu[les.ioutnum,:]
 
        try:
            for _ in range(nIntermediate):
                les.step(actions)

            les.compute_Ek()
            les.fou2real()
        except Exception as e:
            print("Exception occured in LES:")
            print(str(e))
            s["State"] = les.getState().flatten().tolist()
            error = 1
            break
        
        if basereward == False:
            try:
                for _ in range(nIntermediate):
                    vBase = vBase - dt*0.5*les.k1*fft(uBase**2) + dt*nu*les.k2*vBase
                    uBase = np.real(ifft(vBase))
            except Exception as e:
                print("Exception occured in BASE:")
                print(str(e))
                s["State"] = les.getState().flatten().tolist()
                error = 2
                break

        # get new state
        state = les.getState().flatten().tolist()
        if(np.isfinite(state).all() == False):
            s["State"] = state
            print("Nan state detected")
            error = 1
            break

        s["State"] = state
    
        # calculate reward
        uTruth = les.mapGroundTruth()
        
        try:
            uLesDiffMse = ((uTruth[les.ioutnum,:] - les.uu[les.ioutnum,:])**2).mean()
            if basereward == False:
                uBaseDiffMse = ((uTruth[les.ioutnum,:] - uBase)**2).mean()
            else:
                uBaseDiffMse = ((uTruth[les.ioutnum,:] - base0.uu[les.ioutnum,:])**2).mean()
        except Exception as e:
            print("Exception occured in MSE:")
            print(str(e))
            reward = -np.inf
            error = 1
            break
 
        reward = rewardFactor*(uBaseDiffMse-uLesDiffMse)
       
        cumreward += reward
        rewardHistory.append(reward)

        if (np.isfinite(reward) == False):
            print("Nan reward detected")
            error = 1
            break
    
        else:
            s["Reward"] = reward
 
        step += 1

        s["State"] = state
    
    if error == 1:
        s["Termination"] = "Truncated"
        s["Reward"] = -1000 if testing else -np.inf
    
    elif error == 2:
        s["Termination"] = "Truncated"
        s["Reward"] = max(rewardHistory)

    else:
        s["Termination"] = "Terminal"

    print(cumreward)
    if testing:
            
        les.compute_Ek()

        fileName = s["Custom Settings"]["Filename"]
        actionHistory = np.array(actionHistory)
        print("Storing les to file {}".format(fileName))
        np.savez(fileName, x = les.x, t = les.tt, uu = les.uu, vv = les.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=tEnd, actions=actionHistory)
         
        print("Running uncontrolled SGS..")
        base = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
        base.IC(u0 = f_restart(base.x))
        base.simulate()
        base.fou2real()
        base.compute_Ek()
       
        k1 = dns.k[:N//2]

        time = np.arange(tEnd/dt+1)*dt

        fig, axs = plt.subplots(3, 6, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(15,15))

        umax = max(dns.uu.max(), base.uu.max(), les.uu.max())
        umin = min(dns.uu.min(), base.uu.min(), les.uu.min())
        ulevels = np.linspace(umin, umax, 50)

#------------------------------------------------------------------------------
        print("plot DNS")

        axs[0,0].contourf(dns.x, dns.tt, dns.uu, ulevels)

        axs[0,2].plot(time, dns.Ek_t)
        axs[0,2].plot(time, dns.Ek_tt)

        axs[0,4].plot(k1, np.abs(dns.Ek_ktt[-1,0:N//2]),'b')
        axs[0,4].set_xscale('log')
        axs[0,4].set_yscale('log')
 
#------------------------------------------------------------------------------

        errBaseEk_t = dns.Ek_t - base.Ek_t
        errBaseEk_tt = dns.Ek_tt - base.Ek_tt

        f_dns = interpolate.interp2d(dns.x, dns.tt, dns.uu, kind='cubic')
        udns_int = f_dns(base.x, base.tt)
        errBaseU = np.abs(base.uu-udns_int)
        errBaseU_t = np.mean(errBaseU**2, axis=1)

#------------------------------------------------------------------------------

        errEk_t = dns.Ek_t - les.Ek_t
        errEk_tt = dns.Ek_tt - les.Ek_tt
        errU = np.abs(les.uu-udns_int)
        errU_t = np.mean(errU**2, axis=1)
 
        emax = max(errBaseU.max(), errU.max())
        emin = min(errBaseU.min(), errU.min())
        elevels = np.linspace(emin, emax, 50)
        
#------------------------------------------------------------------------------
        print("plot baseline")
        
        k2 = les.k[:gridSize//2]
 
        idx = 1
        # Plot solution
        axs[idx,0].contourf(base.x, base.tt, base.uu, ulevels)
  
        # Plot difference to dns
        axs[idx,1].contourf(les.x, base.tt, errBaseU, elevels)

        # Plot instanteneous energy and time averaged energy
        axs[idx,2].plot(time, base.Ek_t)
        axs[idx,2].plot(time, base.Ek_tt)
     
        # Plot energy differences
        axs[idx,3].plot(time, errBaseEk_t)
        axs[idx,3].plot(time, errBaseEk_tt)
        axs[idx,3].plot(time, errBaseU_t)

        # Plot energy spectrum at start, mid and end of simulation
        axs[idx,4].plot(k2, np.abs(base.Ek_ktt[-1,0:gridSize//2]),'b')
        axs[idx,4].set_xscale('log')
        axs[idx,4].set_yscale('log')
  
        # Plot energy spectrum difference
        axs[idx,4].plot(k2, np.abs(dns.Ek_ktt[-1,0:gridSize//2] - base.Ek_ktt[-1,0:gridSize//2]),'--r')

#------------------------------------------------------------------------------
        print("plot les")
        
        idx += 1
        # Plot solution
        axs[idx,0].contourf(les.x, les.tt, les.uu, ulevels)
        
        # Plot difference to dns
        axs[idx,1].contourf(les.x, les.tt, errU, elevels)
 

        # Plot instanteneous energy and time averaged energy
        axs[idx,2].plot(time, les.Ek_t)
        axs[idx,2].plot(time, les.Ek_tt)
     
        # Plot energy differences
        axs[idx,3].plot(time, errEk_t)
        axs[idx,3].plot(time, errEk_tt)
        axs[idx,3].plot(time, errU_t)

        # Plot energy spectrum at start, mid and end of simulation
        axs[idx,4].plot(k2, np.abs(les.Ek_ktt[-1,0:gridSize//2]),'b')
        axs[idx,4].set_xscale('log')
        axs[idx,4].set_yscale('log')
  
        # Plot energy spectrum difference
        axs[idx,4].plot(k2, np.abs(dns.Ek_ktt[-1,0:gridSize//2] - les.Ek_ktt[-1,0:gridSize//2]),'--r')
 
        # Plot energy spectrum at start, mid and end of simulation
        colors = plt.cm.coolwarm(np.linspace(0,1,numActions))
        for i in range(numActions):
            axs[idx,5].plot(timestamps, actionHistory[:,i], color=colors[i])

        figName = fileName + ".png"
        fig.savefig(figName)

#------------------------------------------------------------------------------

        figName2 = fileName + "_evolution.png"
        print("Plotting {} ...".format(figName2))
        
        fig2, axs = plt.subplots(4,4, sharex=True, sharey=False, figsize=(15,15))
        for i in range(16):
            t = i * tEnd / 16
            tidx = int(t/dt)
            k = int(i / 4)
            l = i % 4
            
            axs[k,l].plot(dns.x, dns.uu[tidx,:], '--k')
            axs[k,l].plot(les.x, les.uu[tidx,:], '-r')
            axs[k,l].plot(base.x, base.uu[tidx,:], '-b')

        fig2.savefig(figName2)
