from KS import *

def environment( args, s ):
    L    = 22/(2*np.pi)
    N    = 64
    dt   = 0.25
    tEnd = 50 #50000
    case = args["case"]
    sim  = KS(L=L, N=N, dt=dt, tend=tEnd, RL=True, case=case)

    # simulate up to T=20
    tInit = 20
    nInitialSteps = int(tInit/dt)
    sim.simulate( nsteps=nInitialSteps )

    ## get initial state
    s["State"] = sim.state().tolist()
    # print("state:", sim.state())

    ## run controlled simulation
    nContolledSteps = int((tEnd-tInit)/dt)
    step = 0
    while step < nContolledSteps:
        # Getting new action
        s.update()

        # apply action and advance environment
        sim.step( s["Action"] )
        # print("action:", s["Action"])

        # get reward
        s["Reward"] = sim.reward()
        # print("state:", sim.reward())

        # get new state
        s["State"] = sim.state().tolist()
        # print("state:", sim.state())
            
        step += 1

    # TODO?: Termination in case of divergence
    s["Termination"] = "Truncated"