from Burger import *
import matplotlib.pyplot as plt 

# dns defaults
N    = 512
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.01

# reward defaults
rewardFactor = 1.

# basis defaults
basis = 'hat'

def fBurger( s , gridSize, episodeLength, ic ):
 
    noisy = False
    
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noisy=noisy)
    dns.simulate()
    dns.fou2real()
    dns.compute_Ek()

    ## create interpolated IC
    f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

    # Initialize LES
    les = Burger(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd, noisy=False)
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

                idx = les.ioutnum
                u = les.uu[les.ioutnum,:] 
                um = np.roll(u, 1)
                up = np.roll(u, -1)
                
                dudx = (u - um)/dx
                dudxx = (up - 2*u + um)/dx
            
                sgs = -cs*cs*dx2*(np.abs(dudxx)*dudx+np.abs(dudx)*dudxx)

                les.step(-sgs)

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

        idx = les.ioutnum
        uTruthToCoarse = les.mapGroundTruth()
        uDiffMse = ((uTruthToCoarse[idx,:] - les.uu[idx,:])**2).mean()
 
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
