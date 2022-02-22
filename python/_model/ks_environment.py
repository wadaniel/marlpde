from KS import *
  
N    = 1024
L    = 22/(2*np.pi)
nu   = 1.0
dt   = 0.01
tTransient = 20
tEnd = 50
tSim = tEnd - tTransient
nSimSteps = tSim/dt

rewardFactor = 1e4

# DNS baseline
dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient)
#dns.simulate()
#dns.fou2real()
 
## restart
#v_restart = dns.vv[-1,:].copy()
#u_restart = dns.uu[-1,:].copy()
 
# set IC
#dns.IC( v0 = v_restart )

# continue simulation
#dns.simulate( nsteps=int(tSim/dt), restart = True )

# convert to physical space
#dns.fou2real()

# calcuate energies
#dns.compute_Ek()

def environment( s , gridSize, numActions, episodeLength ):
 
    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')
 
    # Initialize LES
    les = KS(L=L, N = gridSize, dt=dt, nu=1.0, tend=tEnd-tTransient)
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
