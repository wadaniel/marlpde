from KS import *

def environment( s , nagents ):

    # defaults
    L    = 22/(2*np.pi)
    dt   = 0.1
    tEnd = 50 #50000

    # DNS baseline
    dns = KS(L=L, N=512, dt=dt, nu=1.0, tend=tEnd)
    dns.simulate()
    dns.fou2real()
    
    uTruth = dns.uu
    tTruth = dns.tt
    xTruth = dns.x

    sgs = KS(L=L, N=nagents, dt=dt, nu=1.0, tend=tEnd)
    sgs.setGroundTruth(tTruth, xTruth, uTruth)

    ## create interpolated IC
    u_restart = dns.uu[0,:].copy()
    f_restart = interpolate.interp1d(xTruth, u_restart)

    xCoarse = sgs.x
    uRestartCoarse = f_restart(xCoarse)
    sgs.IC( u0 = uRestartCoarse )

    ## get initial state
    s["State"] = sgs.getState().tolist()
    # print("state:", sim.state())

    ## run controlled simulation
    error = 0
    step = 0
    while step < int(tEnd/dt) and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
        actions = [ a[0] for a in actions ]
        sgs.updateField( actions )
        sgs.step()
        
        # get new state
        s["State"] = sgs.getState().tolist()

        if error == 0:
            # get reward
            s["Reward"] = sgs.getReward().tolist()
            # print("state:", sim.reward())
        else:
            s["Reward"] = -1000
            
        step += 1

    s["Termination"] = "Truncated"


