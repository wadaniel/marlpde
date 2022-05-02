#!/bin/python3

""" Script to test absis function """

import sys
sys.path.append('./../../_model/')

from KS_jax import *
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, random
from numpy.linalg import norm

#!/bin/python3
from KS_jax import *

N    = 1024
N_sgs = 64
L    = 2*np.pi
nu   = 1.0
dt   = 0.25
tTransient = 50
tEnd = 550
tSim = tEnd - tTransient
nSimSteps = int(tSim/dt)
episodeLength = 500

#------------------------------------------------------------------------------
# action defaults
basis = 'hat'
M = 16

# Initialize LES
# simulate transient period
dns = KS_jax(L=L, N=N, dt=dt, nu=nu, tend=tTransient)
dns.simulate()
dns.fou2real()
u_restart = dns.uu[-1,:].copy()
v_restart = dns.vv[-1,:].copy()

# simulate rest
dns.IC( u0 = u_restart)
dns.simulate( nsteps=int(tSim/dt), restart=True )
dns.fou2real()
dns.compute_Ek()

u_restart = dns.uu[0,:].copy()
v_restart = dns.vv[0,:].copy()

# reward defaults
rewardFactor = 1.

## create interpolated IC
f_restart = interpolate.interp1d(dns.x, dns.u0, kind='cubic')

# Initialize LES
sgs = KS_jax(L=L, N = N_sgs, dt=dt, nu=nu, tend=tSim)
v0 = np.concatenate((v_restart[:((N_sgs+1)//2)], v_restart[-(N_sgs-1)//2:])) * N_sgs / dns.N

sgs.IC( v0 = v0 )
sgs.setup_basis(M, basis)
## run controlled simulation
step = 0
nIntermediate = int(tSim / dt / episodeLength)
#nIntermediate = 10
error = 0
while step < episodeLength and error == 0:

    # apply action and advance environment
    actions = jnp.asarray(np.random.normal(loc=0., scale=1e-2, size=M))
    print(step)
    try:
        sgs.step(actions, nIntermediate)
    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break
    #print(sgs.u)
    #print("field")
    #print(sgs.u)
    #print("grads")
    #print(sgs.gradient)
    step += 1
