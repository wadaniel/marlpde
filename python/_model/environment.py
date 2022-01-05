from KS import *
  
# defaults
L    = 22/(2*np.pi)
dt   = 0.1
tTransient = 10
tEnd = 50 #50000
tSim = tEnd - tTransient
   
# DNS baseline
dns = KS(L=L, N=512, dt=dt, nu=1.0, tend=tTransient)
dns.simulate()
dns.fou2real()
  
## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()
 
def environment( s , nagents ):
  
    # continue simulation
    dns.simulate( nsteps=int(tSim/dt), restart=True )

    # convert to physical space
    dns.fou2real()

    #uTruth = dns.uu
    #tTruth = dns.tt
    #xTruth = dns.x

    les = KS(L=L, N=nagents, dt=dt, nu=1.0, tend=tEnd-tTransient)
    #les.setGroundTruth(tTruth, xTruth, uTruth)

    ## create interpolated IC
    f_restart = interpolate.interp1d(xTruth, u_restart)

    xLES = les.x
    uRestartLES = f_restart(xLES)
    les.IC( u0 = uRestartLES )

    ## get initial state
    s["State"] = les.getState().tolist()

    ## run controlled simulation
    error = 0
    step = 0
    while step < int(tEnd/dt) and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
        actions = [ a[0] for a in actions ]
        les.updateField( actions )
        try:
            les.step()
        except Exception as e:
            print(e.str())
            error = 1
            break
        
        # get new state
        state = les.getState()
        isnan = np.isnan(state).any()
        if(isnan == True):
            print("Nan state detected")
            error = 1
            break
 
        s["State"] = state.tolist()

        # calculate energy
        dnsEnergy = 0.5*np.sum(dns.uu[step,:]**2)*dns.dx
        lesEnergy = 0.5*np.sum(les.uu[step,:]**2)*les.dx

        # calculate reward from energy
        s["Reward"] = [abs(dnsEnergy - lesEnergy)]*nagents
        
        
        #s["Reward"] = les.getReward().tolist()
        step += 1

        
    if error == 1:
        s["Termination"] = "Truncated"
        s["Reward"] = [-3000] * nagents
    
    else:
        s["Termination"] = "Terminal"
            



