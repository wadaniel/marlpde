from KS import *
from KS_clos import *
import matplotlib.pyplot as plt

# dns defaults
N    = 1024
L    = 22
nu   = 1.0
dt   = 0.25
tTransient = 50
tEnd = 550
tSim = tEnd - tTransient
nSimSteps = int(tSim/dt)
basis = 'hat'

def setup_dns_default(N, dt, nu , seed):
    print("Setting up default dns with args ({}, {}, {}, {})".format(N, dt, nu, seed))
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

def fKS( s , N, gridSize, dt, nu, episodeLength, spectralReward, noise, seed, ssm, dsm, dns_default = None ):

    dns = dns_default

    u_restart = dns.uu[0,:].copy()
    v_restart = dns.vv[0,:].copy()

    # reward defaults
    rewardFactor = 1.

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    les = KS_clos(L=L, N = gridSize, dt=dt, nu=nu, tend=tSim, ssm=ssm, dsm=dsm, noise=0.)
    v0 = np.concatenate((v_restart[:((gridSize+1)//2)], v_restart[-(gridSize-1)//2:])) * gridSize / dns.N

    les.IC( v0 = v0 )
    les.setGroundTruth(dns.tt, dns.x, dns.uu)

    ## get initial state
    cs = s["Parameters"][0]
    print("Param: {}".format(cs))

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tSim / dt / episodeLength)
    cumreward = 0.

    while step < episodeLength and error == 0:

        try:
            for _ in range(nIntermediate):

                les.step(cs)

            les.compute_Ek()

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

        # calculate reward
        if spectralReward:
            #kMseLogErr = np.mean((np.log(dns.Ek_ktt[les.ioutnum,:gridSize]) - np.log(les.Ek_ktt[les.ioutnum,:gridSize]))**2)
            kMseLogErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,:gridSize//2] - les.Ek_ktt[sgs.ioutnum,:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,:gridSize//2])**2)
            reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
            prevkMseLogErr = kMseLogErr

        else:
            uTruthToCoarse = les.mapGroundTruth()
            uDiffMse = ((uTruthToCoarse[les.ioutnum-nIntermediate:les.ioutnum,:] - les.uu[les.ioutnum-nIntermediate:les.ioutnum,:])**2).mean()
            reward = -rewardFactor*uDiffMse

        cumreward += reward

        if (np.isfinite(reward) == False):
            print("Nan reward detected")
            error = 1
            break

        step += 1

    if error == 1:
        cumreward = -1e6

    print("Obj: {}".format(cumreward))
    s["F(x)"] = cumreward
