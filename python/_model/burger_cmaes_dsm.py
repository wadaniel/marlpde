from Burger import *
import matplotlib.pyplot as plt

# dns defaults
L    = 2*np.pi
tEnd = 5
basis = 'hat'

def setup_dns_default(N, dt, nu , ic, seed):
    print("Setting up default dns with args ({}, {}, {}, {}, {})".format(N, dt, nu, ic, seed))
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=0., seed=seed)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()
    return dns

def fBurger( s , N, gridSize, dt, nu, episodeLength, ic, spectralReward, noise, seed, dns_default = None ):

    if noise > 0.:
        dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed)
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
    les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noise=0.)
    if spectralReward:
        v0 = np.concatenate((dns.v0[:((gridSize+1)//2)], dns.v0[-(gridSize-1)//2:]))
        les.IC( v0 = v0 * gridSize / N )
    else:
        les.IC( u0 = f_restart(les.x) )

    les.setup_basis(gridSize, basis)
    les.setGroundTruth(dns.tt, dns.x, dns.uu)

    ## get initial state

    # determien box filter
    # Gbox = np.zeros(gridSize)
    # dx_ = 2*les.dx # test scale width
    # for i in range(gridSize):
    #     val = i*les.dx
    #     if abs(val) <= les.dx:
    #         Gbox[i] = 1/(dx_)
    #     else:
    #         Gbox[i] = 0

    # determine sharp spectral filter
    Gspec = np.zeros(gridSize)
    dx_ = 2*les.dx # test scale width
    for i in range(gridSize):
        val = i*les.dx
        Gspec[i] = np.sin(np.pi*val/dx_)/(np.pi*val)

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

                idx = les.ioutnum
                u = les.uu[les.ioutnum,:]
                um = np.roll(u, 1)
                up = np.roll(u, -1)

                dudx = (u - um)/dx
                d2udx2 = (up - 2*u + um)/dx2

                #find C via germano identity, no averaging
                # u_ = np.convolve(u, Gbox, mode = 'same')
                # u2_ = np.convolve(u*u, Gbox, mode = 'same')
                # L_ = u2_ - u_
                # #
                # dudx_ = np.convolve(dudx, Gbox, mode = 'same')
                # dudx_abs = np.absolute(dudx_)
                # M = dx*dx*np.convolve(np.absolute(dudx)*dudx, Gbox, mode = 'same') - dx_*dx_*dudx_abs*dudx_
                #
                # C = L*M*np.reciprocal(M*M)/2

                #find C via germano identity, with averaging
                u_ = np.convolve(u, Gbox)
                u2_ = np.convolve(u*u, Gbox)
                L_ = u2_ - u_

                dudx_ = np.convolve(dudx, Gbox)
                dudx_abs = np.absolute(dudx_)
                M = dx*dx*np.convolve(np.absolute(dudx)*dudx, Gbox) - dx_*dx_*dudx_abs*dudx_

                C = np.mean(L*M)/(2*np.mean(M*M))

                sgs = 2*C*C*dx2*(d2udx2)*(dudx**2)/(np.absolute(dudx))
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
