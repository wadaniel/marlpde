from Burger import *
import matplotlib.pyplot as plt 

N    = 1024
L    = 2*np.pi
dt   = 0.0005
tEnd = 5
nu   = 0.01

# reward defaults
rewardFactor = 1.

# DNS baseline
print("Setting up DNS..")
dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd)
dns.simulate()
dns.fou2real()
dns.compute_Ek()

## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

# calcuate energies
tAvgEnergy = dns.Ek_tt
print("Done!")

def environment( s , gridSize, episodeLength ):
 
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False

    # Initialize LES
    les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd)
    les.IC( u0 = f_restart(les.x) )

    ## get initial state
    state = les.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tEnd / dt / episodeLength)
    cumreward = 0.
    while step < episodeLength and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
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
        tAvgEnergyLES = les.Ek_tt
        reward = -rewardFactor*(np.abs(tAvgEnergyLES[step*nIntermediate]-tAvgEnergy[step*nIntermediate]))
        cumreward += reward

        if (np.isnan(reward)):
            print("Nan reward detected")
            error = 1
            break

        s["Reward"] = reward
        step += 1

    print(cumreward)
    if error == 1:
        s["State"] = state
        s["Termination"] = "Truncated"
        s["Reward"] = -500
    
    else:
        s["Termination"] = "Terminal"

    if testing:

        fileName = s["Custom Settings"]["Filename"]
        print("Storing les to file {}".format(fileName))
        np.savez(fileName, x = les.x, t = les.tt, uu = les.uu, vv = les.vv, L=L, N=gridsize, dt=dt, nu=nu, tEnd=tEnd)
         
        print("Setting up DNS..")
        base = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd)
        base.simulate()
        base.fou2real()
        base.compute_Ek()
       
        print("plot DNS")
        k1 = dns.k[:N//2]

        time = np.arange(tEnd/dt+1)*dt
        s, n = np.meshgrid(2*np.pi*L/N*(np.array(range(N))+1), time)

        fig, axs = plt.subplots(2, 5, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(15,15))
        axs[0,0].contourf(s, n, dns.uu, 50)

        axs[0,2].plot(time, dns.Ek_t)
        axs[0,2].plot(time, dns.Ek_tt)

        axs[0,4].plot(k1, np.abs(dns.Ek_ktt[0,0:N//2]),'b:')
        axs[0,4].plot(k1, np.abs(dns.Ek_ktt[tEnd//2,0:N//2]),'b--')
        axs[0,4].plot(k1, np.abs(dns.Ek_ktt[-1,0:N//2]),'b')
        axs[0,4].set_xscale('log')
        axs[0,4].set_yscale('log')

#------------------------------------------------------------------------------
        errEk_t = dns.Ek_t - les.Ek_t
        errEk_tt = dns.Ek_tt - les.Ek_tt

        f_dns = interpolate.interp2d(dns.x, dns.tt, dns.uu, kind='cubic')
        udns_int = f_dns(les.x, les.tt)
        errU = np.abs(les.uu-udns_int)
#------------------------------------------------------------------------------
  
        k2 = les.k[:gridSize//2]
        s, n = np.meshgrid(2*np.pi*L/gridSize*(np.array(range(gridSize))+1), time)

        idx = 1
        # Plot solution
        axs[idx,0].contourf(s, n, les.uu, 50)
        
        # Plot difference to dns
        axs[idx,1].contourf(les.x, les.tt, errU, 50)

        # Plot instanteneous energy and time averaged energy
        axs[idx,2].plot(time, les.Ek_t)
        axs[idx,2].plot(time, les.Ek_tt)
     
        # Plot energy differences
        axs[idx,3].plot(time, errEk_t)
        axs[idx,3].plot(time, errEk_tt)

        # Plot energy spectrum at start, mid and end of simulation
        axs[idx,4].plot(k2, np.abs(les.Ek_ktt[0,0:gridSize//2]),'b:')
        axs[idx,4].plot(k2, np.abs(les.Ek_ktt[tEnd//2,0:gridSize//2]),'b--')
        axs[idx,4].plot(k2, np.abs(les.Ek_ktt[-1,0:gridSize//2]),'b')
        axs[idx,4].set_xscale('log')
        axs[idx,4].set_yscale('log')

        fig.savefig('rl_les.png')
