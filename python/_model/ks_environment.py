from KS import *
from plotting import *
import matplotlib.pyplot as plt

N    = 1024
L    = 22
nu   = 1.0
dt   = 0.25
tTransient = 50
tEnd = 550
tSim = tEnd - tTransient
nSimSteps = int(tSim/dt)
basis = 'hat'

errModes = 4

# DNS baseline
def setup_dns_default(N, dt, nu , seed):
    print("[ks_environment] setting up default dns")

    # simulate transient period
    dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient, seed=seed)
    dns.simulate()
    dns.fou2real()
    u_restart = dns.uu[-1,:].copy()
    v_restart = dns.vv[-1,:].copy()

    # simulate rest
    dns.IC( u0 = u_restart)
    dns.simulate( nsteps=int(tSim/dt), restart=True )
    dns.fou2real()
    dns.compute_Ek()

    return dns

def environment( s , N, gridSize, numActions, dt, nu, episodeLength, dforce, seed, dns_default):
    
    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False
    dns = dns_default

    u_restart = dns.uu[0,:].copy()
    v_restart = dns.vv[0,:].copy()
 
    # reward defaults
    rewardFactor = 1.

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    sgs = KS(L=L, N = gridSize, dt=dt, nu=nu, tend=tSim, dforce=dforce, noise=0.)
    v0 = np.concatenate((v_restart[:((gridSize+1)//2)], v_restart[-(gridSize-1)//2:])) * gridSize / dns.N
    
    sgs.IC( v0 = v0 )
    sgs.setup_basis(numActions, basis)

    ## get initial state
    state = sgs.getState().flatten().tolist()
    s["State"] = state

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tSim / dt / episodeLength)
    prevkMseLogErr = 0.
    kMseLogErr = 0.
    reward = 0.
    cumreward = 0.

    while step < episodeLength and error == 0:
        
        # Getting new action
        s.update()

        # apply action and advance environment
        actions = s["Action"]

        try:
            for _ in range(nIntermediate):
                sgs.step(actions)
            sgs.compute_Ek()
            sgs.fou2real()

        except Exception as e:
            print("[ks_environment] Exception occured:")
            print(str(e))
            error = 1
            break
        
        # get new state
        state = sgs.getState().flatten().tolist()
        if(np.isnan(state).any() == True):
            print("[ks_environment] Nan state detected")
            error = 1
            break
        s["State"] = state
 
        kMseLogErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,1:gridSize//2] - sgs.Ek_ktt[sgs.ioutnum,1:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,1:gridSize//2])**2)
        reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
        prevkMseLogErr = kMseLogErr

        cumreward += reward
        
        if (np.isnan(reward)):
            print("[ks_environment] Nan reward detected")
            error = 1
            break

        s["Reward"] = reward
        step += 1

    print(cumreward)

    if error == 1:
        s["Termination"] = "Truncated"
        s["Termination"] = "Truncated"
        s["Reward"] = -1000 if testing else -np.inf
    
    else:
        s["Termination"] = "Terminal"
    
    if testing:

        # store controlled LES
        fileName = s["Custom Settings"]["Filename"]
        print("[ks_environment] Storing sgs to file {}".format(fileName))
        np.savez(fileName, x = sgs.x, t = sgs.tt, uu = sgs.uu, vv = sgs.vv, L=L, N=gridSize, dt=dt, nu=nu, tEnd=tSim)
         
        print("[ks_environmnet] Calculating SGS terms from DNS..")
        dns.compute_Sgs(gridSize)
  
        print("[ks_environment] Running UGS..")
        base = KS(L=L, N = gridSize, dt=dt, nu=nu, tend=tSim, dforce=dforce, noise=0.)
        v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        base.IC( v0 = v0 * gridSize / dns.N )
        base.setup_basis(numActions, basis)
        base.setGroundTruth(dns.tt, dns.x, dns.uu)

        # reinit vars
        error = 0
        step = 0
        prevkMseLogErr = 0.
        kMseLogErr = 0.
        reward = 0.
        cumreward = 0.

        actions = np.zeros(numActions)

        while step < episodeLength and error == 0:
        
            # apply action and advance environment
            try:
                for _ in range(nIntermediate):
                    base.step(actions)

                base.compute_Ek()
                base.fou2real()
            except Exception as e:
                print("[ks_environment] Exception occured:")
                print(str(e))
                error = 1
                break
            

            # calculate reward
            kMseLogErr = np.mean((np.abs(dns.Ek_ktt[base.ioutnum,:gridSize//2] - base.Ek_ktt[base.ioutnum,:gridSize//2])/dns.Ek_ktt[base.ioutnum,:gridSize//2])**2)
            reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
            prevkMseLogErr = kMseLogErr

            # accumulat reward
            cumreward += reward

            if (np.isfinite(reward) == False):
                print("[ks_environment] Nan reward detected")
                error = 1
                break
     
            step += 1

        print("[ks_environment] uncontrolled cumreward")
        print(cumreward)
        
        makePlot(dns, base, sgs, fileName, True)
