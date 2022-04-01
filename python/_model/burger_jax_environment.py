from Burger_jax import *
from plotting import *

# dns defaults
L    = 2*np.pi
tEnd = 5
basis = 'hat'

def setup_dns_default(N, dt, nu , ic, seed):
    print("[burger_jax_env] Setting up default dbs with args ({}, {}, {}, {}, {})".format(N, dt, nu, ic, seed))
    dns = Burger_jax(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=0., seed=seed)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()
    return dns
 
def environment( s , N, gridSize, numActions, dt, nu, episodeLength, ic, spectralReward, dforce, noise, seed, dns_default = None ):

    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    noise = 0. if testing else noise

    if noise > 0.:
        dns = Burger_jax(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, dforce=dforce, noise=noise, seed=seed)
        dns.simulate()
        dns.fou2real()
        dns.compute_Ek()
    else:
        dns = dns_default
 
    # reward defaults
    rewardFactor = 1. if spectralReward else 1.

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    sgs = Burger_jax(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
    if spectralReward:
        v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        sgs.IC( v0 = v0 * gridSize / dns.N )
    else:
        sgs.IC( u0 = f_restart(sgs.x) )

    sgs.setup_basis(numActions, basis)
    sgs.setGroundTruth(dns.tt, dns.x, dns.uu)

    ## get initial state
    state = sgs.getState().flatten().tolist()
    s["State"] = state
    s["State Gradient"] = sgs.getGrad().tolist()
    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tEnd / dt / episodeLength)
 
    prevkMseLogErr = 0.
    kMseLogErr = 0.
    reward = 0.
    cumreward = 0.

    timestamps = []
    actionHistory = []

    while step < episodeLength and error == 0:
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]
        
        try:
            sgs.step(actions, nIntermediate)
            sgs.compute_Ek()
            sgs.fou2real()
        except Exception as e:
            print("[burger_jax_env] Exception occured during in stepping..:")
            print(str(e))
            error = 1
            break


        # get new state
        newstate = sgs.getState().flatten().tolist()
        newgrad = sgs.getGrad()
        if(np.isfinite(newstate).all() == False and np.isfinite(newgrad).all() == False):
            print("[burger_jax_env] Nan state detected")
            error = 1
            break
        else:
            state = newstate
            gradient = newgrad

        s["State"] = state
        s["State Gradient"] = gradient.tolist()
        
        # calculate reward
        if spectralReward:
            kMseLogErr = np.mean((np.log(dns.Ek_ktt[sgs.ioutnum,:gridSize]) - np.log(sgs.Ek_ktt[sgs.ioutnum,:gridSize]))**2)
            reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
            prevkMseLogErr = kMseLogErr

        else:
            uTruthToCoarse = sgs.mapGroundTruth()
            uDiffMse = ((uTruthToCoarse[sgs.ioutnum-nIntermediate:sgs.ioutnum,:] - sgs.uu[sgs.ioutnum-nIntermediate:sgs.ioutnum,:])**2).mean()
            reward = -rewardFactor*uDiffMse

        cumreward += reward

        if (np.isfinite(reward) == False):
            print("[burger_jax_env] Nan reward detected")
            error = 1
            break

        else:
            s["Reward"] = reward

        step += 1

    print(cumreward)
    if error == 1:
        s["State"] = state
        s["State Gradient"] = sgs.getGrad().tolist()
        s["Termination"] = "Truncated"
        s["Reward"] = -1000 if testing else -np.inf

    else:
        s["Termination"] = "Terminal"

    if testing:

        fileName = s["Custom Settings"]["Filename"]
        actionHistory = np.array(actionHistory)
        print("[burger_jax_env] Storing sgs to file {}".format(fileName))
        np.savez(fileName, x = sgs.x, t = sgs.tt, uu = sgs.uu, vv = sgs.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=tEnd, actions=actionHistory)

        print("[burger_jax_env] Running uncontrolled SGS..")
        base = Burger_jax(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
        if spectralReward:
            print("[burger_jax_env] Init spectrum.")
            v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
            base.IC( v0 = v0 * gridSize / dns.N )

        else:
            print("[burger_jax_env] Init interpolation.")
            base.IC( u0 = f_restart(base.x) )

        base.simulate()
        base.fou2real()
        base.compute_Ek()

        makePlot(dns, base, sgs, fileName)
