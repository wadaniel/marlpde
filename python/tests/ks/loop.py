#!/bin/python3
import sys
sys.path.append('./../../_model/')
from KS import *
 
import matplotlib.pyplot as plt

# LES defaults
numActions = 1
numGridPoints = 32
 
# DNS defaults
N    = 1024
L    = 22/(2*np.pi)
nu   = 1.0
dt   = 0.01
tTransient = 50
tEnd = 100
tSim = tEnd - tTransient
nSimSteps = tSim/dt

rewardFactor = 10
  
# DNS baseline
dns = KS(L=L, N=N, dt=dt, nu=nu, tend=tTransient)
dns.simulate()
dns.fou2real()
  
## restart
v_restart = dns.vv[-1,:].copy()
u_restart = dns.uu[-1,:].copy()
 
## create interpolated IC
f_restart = interpolate.interp1d(dns.x, u_restart, kind='cubic')

# set IC
dns.IC( v0 = v_restart )

# continue simulation
dns.simulate( nsteps=int(tSim/dt), restart = True )

# convert to physical space
dns.fou2real()

# calcuate energies
dns.compute_Ek()
tAvgEnergy = dns.Ek_tt


# init rewards
rewards = []

# Initialize LES
les = KS(L=L, N = numGridPoints, dt=dt, nu=nu, tend=tEnd-tTransient)
les.IC( u0 = f_restart(les.x) )
les.setup_basis(numActions)

## run controlled simulation
error = 0
step = 0
while step < int(tSim/dt) and error == 0:
    
    actions = [0.]
    les.step(actions)
    les.compute_Ek()
    
    tAvgEnergyLES = les.Ek_tt

    # calculate reward from energy
    reward = -rewardFactor*(np.abs(tAvgEnergyLES[step] -tAvgEnergy[step])) #**2
    rewards.append(reward)
    
    step += 1

rewards = np.array(rewards)
cumRewards = np.cumsum(rewards)

## plot rewards
time = np.arange(tSim/dt)*dt
fig, axs = plt.subplots(1,2)

axs[0].plot(time, -rewards)
axs[0].set_yscale('log')
axs[1].plot(time, cumRewards)
fig.suptitle("Neg log-rewards and cumulative rewards")
plt.savefig('loop.png')

print("Cum Reward {}".format(cumRewards[-1]))
