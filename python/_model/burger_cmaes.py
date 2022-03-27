from Burger import *
import matplotlib.pyplot as plt 

# dns defaults
N    = 512
L    = 2*np.pi
dt   = 0.001
tEnd = 5
nu   = 0.01

# reward structure
spectralReward = True

# reward defaults
rewardFactor = 0.001 if spectralReward else 1.

# basis defaults
basis = 'hat'

def fBurger( s , gridSize, episodeLength, ic, seed):
 
    noisy = False
    
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noisy=noisy, seed=seed)
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
                d2udx2 = (up - 2*u + um)/dx2

                absolute = np.ones(gridSize)
                absolute[dudx<0.] = -1.

                #sgs = cs*cs*dx2*(d2udx2*dudx+dudx*d2udx2)*absolute
                sgs = cs*cs*dx2*(d2udx2*d2udx2)
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
 
        if spectralReward:
            # Time-averaged energy spectrum as a function of wavenumber
            #kMseErr = np.mean((dns.Ek_ktt[les.ioutnum,:gridSize] - les.Ek_ktt[les.ioutnum,:gridSize])**2)
            kMseLogErr = np.mean((np.log(dns.Ek_ktt[les.ioutnum,:gridSize]) - np.log(les.Ek_ktt[les.ioutnum,:gridSize]))**2)
            reward = -rewardFactor*kMseLogErr + 3.5/500
 
        else:
            reward = rewardFactor*les.getMseReward()
 
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
