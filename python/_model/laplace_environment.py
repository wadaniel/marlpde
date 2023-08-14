from Laplace import *
import plotting_laplace as lplt

import matplotlib.pyplot as plt 

# defaults
NDNS=512
L = 2*np.pi

# basis defaults
basis = 'hat'

def environment( s , N, dtSgs, episodeLength, ic, sforce, noise= 0., numAgents=1, version=0):

    testing = True if s["Custom Settings"]["Mode"] == "Testing" else False

    les = Laplace(L=L, N=N, dt=dtSgs, ic=ic, sforce=sforce, noise=noise, episodeLength=episodeLength, version=version)

    step = 0
    stop = False
    cumreward = 0.

    state = les.getState(numAgents)
    s["State"] = state

    while step < episodeLength and stop == False:
    
        # Getting new action
        s.update()
       
        # apply action and advance environment
        actions = s["Action"]

        les.step(actions,numAgents)
        
        reward = les.getDirectReward(numAgents)

        state = les.getState(numAgents)

        s["State"] = state
        s["Reward"] = reward
        
        step += 1
        cumreward += reward if numAgents == 1 else sum(reward)/numAgents

    print(f"[laplace_environment] steps: {step}, cumreward {cumreward}")
    s["Termination"] = "Terminal" 
    
    if testing:

        lplt.plotEvolution(les)
        lplt.plotActionField(les)
        lplt.plotGradientField(les)
        lplt.plotActionDistribution(les)



 
