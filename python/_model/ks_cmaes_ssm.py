from KS import *
import matplotlib.pyplot as plt

# dns defaults
L    = 2*np.pi
tEnd = 5
basis = 'hat'

def setup_dns_default(N, dt, nu , ic, seed):
    print("Setting up default dns with args ({}, {}, {}, {}, {})".format(N, dt, nu, ic, seed))
    dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()
    return dns

def fKS( s , N, gridSize, dt, nu, episodeLength, ic, spectralReward, noise, seed, dns_default = None ):

    dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()

    # reward defaults
    rewardFactor = 1. if spectralReward else 1.

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    les = KS(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd)
    if spectralReward:
        v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        les.IC( v0 = v0 * gridSize / N )
    else:
        les.IC( u0 = f_restart(les.x) )

    les.setup_basis(gridSize, basis)
    les.setGroundTruth(dns.tt, dns.x, dns.uu)

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

                dx = les.dx
                dx2 = dx*dx
                dx3 = dx2*dx
                dx4 = dx3*dx

                idx = les.ioutnum
                u = les.uu[les.ioutnum,:]
                um = np.roll(u, 1)
                umm = np.roll(u, 2)
                up = np.roll(u, -1)
                upp = np.roll(u, -2)

                dudx = (u - um)/dx
                d2udx2 = (up - 2*u + um)/dx
                d3udx3 = (upp - 2*up + 2*um - umm)/(2*dx3)
                d4udx4 = (upp - 4*up + 6*u - 4*um + umm)/dx4

                #sgs = 2*cs*cs*dx2*(d2udx2)*(dudx**2)/(np.absolute(dudx))
                #sgs = 2*cs*cs*dx2*(d4udx4*np.absolute(dudx) + d4udx4*dudx*d2udx2/(np.absolute(dudx)))
                sgs = 2*cs*cs*dx2*np.absolute(dudx)*d4udx4
                les.step(sgs)

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
            kMseLogErr = np.mean((np.log(dns.Ek_ktt[les.ioutnum,:gridSize]) - np.log(les.Ek_ktt[les.ioutnum,:gridSize]))**2)
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
