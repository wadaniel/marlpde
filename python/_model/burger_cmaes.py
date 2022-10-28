from Burger2 import *
import matplotlib.pyplot as plt

# dns defaults
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.02
nt   = int(tEnd/dt)
ic   = 'sinus'
seed = 42
basis = 'hat'

def setup_dns_default(N, dt, nu , ic, seed):
    print("Setting up default dns with args ({}, {}, {}, {}, {})".format(N, dt, nu, ic, seed))
    dns = Burger2(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, seed=seed)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()
    return dns

def fBurger( s , N, gridSize, dt, nu, episodeLength, ic, spectralReward, noise, seed, dns_default = None ):

    if noise > 0.:
        dns = Burger2(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
        dns.simulate()
        dns.fou2real()
        dns.compute_Ek()
    else:
        dns = dns_default

    # reward defaults
    rewardFactor = 1. if spectralReward else 1.

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    sgs = Burger2(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, ssm=True)
    if spectralReward:
        v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        sgs.IC( v0 = v0 * gridSize / N )
    else:
        sgs.IC( u0 = f_restart(sgs.x) )

    sgs.setup_basis(gridSize, basis)
    sgs.setGroundTruth(dns.tt, dns.x, dns.uu)

    ## get initial state
    cs = s["Parameters"][0]
    print("Param: {}".format(cs))

    ## run controlled simulation
    error = 0
    step = 0
    nIntermediate = int(tEnd / dt / episodeLength)
    cumreward = 0.

    while step < episodeLength and error == 0:

        try:
            for _ in range(nIntermediate):

                sgs.step(C=cs)

            sgs.compute_Ek()

        except Exception as e:
            print("Exception occured:")
            print(str(e))
            error = 1
            break

        # get new state
        newstate = sgs.getStateSSM()
        if(np.isfinite(newstate).all() == False):
            print("Nan state detected")
            error = 1
            break
        else:
            state = newstate

        # calculate reward
        if spectralReward:
            kMseLogErr = np.mean((np.abs(dns.Ek_ktt[sgs.ioutnum,:gridSize//2] - sgs.Ek_ktt[sgs.ioutnum,:gridSize//2])/dns.Ek_ktt[sgs.ioutnum,:gridSize//2])**2)
            reward = rewardFactor*(prevkMseLogErr-kMseLogErr)
            prevkMseLogErr = kMseLogErr


        else:
            uTruthToCoarse = sgs.mapGroundTruth()
            uDiffMse = ((uTruthToCoarse[sgs.ioutnum-nIntermediate:sgs.ioutnum,:] - sgs.uu[sgs.ioutnum-nIntermediate:sgs.ioutnum,:])**2).mean()
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
