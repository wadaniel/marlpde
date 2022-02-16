from Burger import *
  
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
    while step < episodeLength and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
        print(actions)
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

        print(state)
        s["State"] = state

        # calculate reward from energy
        tAvgEnergyLES = les.Ek_tt

        reward = -rewardFactor*(np.abs(tAvgEnergyLES[step]-tAvgEnergy[step]))

        if (np.isnan(reward)):
            print("Nan reward detected")
            error = 1
            break

        print(reward)
        s["Reward"] = reward
        step += 1

        
    if error == 1:
        s["Termination"] = "Truncated"
        s["Reward"] = -100
    
    else:
        s["Termination"] = "Terminal"
