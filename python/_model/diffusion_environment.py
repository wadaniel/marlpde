from Diffusion import *
import matplotlib.pyplot as plt 

# dns defaults
L    = 2*np.pi
dt   = 0.001
tEnd = 5

# reward defaults
rewardFactor = 1e6

# basis defaults
basis = 'hat'

def environment( s , N, dt_sgs, numActions, nu, episodeLength, ic, dforce, noise, seed, nunoise=False, version=0):
    
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    noise = 0.0 #if testing else 0.1

    dns = Diffusion(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, version=version, nunoise=nunoise)
    dns.simulate()

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    les = Diffusion(L=L, N=N, dt=dt_sgs, nu=nu, tend=tEnd, u0 = f_restart(dns.x), noise=0. )
    les.setup_basis(numActions, basis)
    les.setGroundTruth(dns.tt, dns.x, dns.uu)

    ## get initial state
    state = les.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tEnd / dt_sgs / episodeLength)
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
        
        except Exception as e:
            print("Exception occured:")
            print(str(e))
            error = 1
            break
        

        # get new state
        newstate = les.getState().flatten().tolist()
        if(np.isfinite(newstate).all() == False):
            print("Nan state detected")
            error = 1
            break
        else:
            state = newstate

        s["State"] = state
    
        # calculate reward
        reward = rewardFactor*les.getMseReward()
 
        cumreward += reward

        if (np.isfinite(reward) == False):
            print("Nan reward detected")
            error = 1
            break
        else:
            s["Reward"] = reward
        
        step += 1


    print(cumreward)
    if error == 1:
        s["State"] = state
        s["Termination"] = "Truncated"
        s["Reward"] = -np.inf
    
    else:
        s["Termination"] = "Terminal"
    
    if testing:

        print("[diffusion_environment] TODO testing")
