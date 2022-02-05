from KS import *
  
# defaults
L    = 22/(2*np.pi)
dt   = 0.1
tTransient = 20
tEnd = 50 #5000
tSim = tEnd - tTransient
rewardFactor = 1e4

# DNS baseline
dns = KS(L=L, N=512, dt=dt, nu=1.0, tend=tTransient)
dns.simulate()
dns.fou2real()
  
## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()
 
## create interpolated IC
f_restart = interpolate.interp1d(dns.x, u_restart, kind='cubic')

# set IC
dns.IC( v0 = v_restart )

# continue simulation
dns.simulate( nsteps=int(tSim/dt), restart = True )

# convert to physical space
dns.fou2real()

# calcuate energies
dns.compute_Ek()
tAvgEnergy = dns.Ek_tt

def environment( s , numGridPoints ):
  
    # Initialize LES
    les = KS(L=L, N = numGridPoints, dt=dt, nu=1.0, tend=tEnd-tTransient)
    les.IC( u0 = f_restart(les.x))

    ## get initial state
    state = les.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    while step < int(tSim/dt) and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
        try:
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

        reward = -rewardFactor*(np.abs(tAvgEnergyLES[step] -tAvgEnergy[step]))
        if (np.isnan(reward)):
            print("Nan reward detected")
            error = 1
            break

        s["Reward"] = reward
        step += 1

        
    if error == 1:
        s["Termination"] = "Truncated"
        s["Reward"] = -1000
    
    else:
        s["Termination"] = "Terminal"
