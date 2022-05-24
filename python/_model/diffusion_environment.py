from Diffusion import *
import matplotlib.pyplot as plt 

# defaults
L = 2*np.pi

# reward defaults
rewardFactor = 1e6

# basis defaults
basis = 'hat'

def environment( s , N, tEnd, dt_sgs, numActions, nu, episodeLength, ic, dforce, noise, seed, nunoise=False, tnoise=False, version=0):
    
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    noise = 0.0 if testing else 0.1

    # Initialize LES
    les = Diffusion(L=L, N=N, dt=dt_sgs, nu=nu, tend=tEnd, case=ic, version=version, noise=0. )
    les.setup_basis(numActions, basis)

    ## get initial state
    state = les.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tEnd / dt_sgs / episodeLength)
    assert nIntermediate > 0
    cumreward = 0.
    cumMseDiff = 0.

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
        
        except Exception as e:
            print("Exception occured:")
            print(str(e))
            error = 1

        # get new state
        newstate = les.getState().flatten().tolist()
        if(np.isfinite(newstate).all() == False):
            print("Nan state detected")
            error = 1
        else:
            state = newstate

        s["State"] = state
    
        # calculate reward
        sol = les.getAnalyticalSolution(les.t)
        uDiffMse = ((sol - les.uu[les.ioutnum,:])**2).mean()
        cumMseDiff += uDiffMse
        reward = -rewardFactor*uDiffMse
 
        cumreward += reward
        s["Reward"] = reward

        if (np.isfinite(reward) == False):
            print("Nan reward detected")
            error = 1
        
        #if uDiffMse > 1.:
        #    error = 1

        step += 1

    print(step)
    print(cumreward)
    if error == 1:
        s["State"] = state
        s["Termination"] = "Truncated"
        s["Reward"] = reward
    
    else:
        s["Termination"] = "Terminal"
    
    if testing:
        print("[diffusion_environment] TODO testing")
