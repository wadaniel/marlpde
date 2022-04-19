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

N    = 64
L    = 2*np.pi
nu   = 1.0
dt   = 0.001
tEnd = 5
episodeLength = 500

#------------------------------------------------------------------------------
# action defaults
basis = 'hat'
M = 16

# Initialize LES
les = KS_jax(L=L, N=N, dt=dt, nu=nu)
les.setup_basis(M, basis)
## run controlled simulation
step = 0
nIntermediate = int(tEnd / dt / episodeLength)
#nIntermediate = 10
error = 0
while step < episodeLength and error == 0:

    # apply action and advance environment
    actions = jnp.asarray(np.random.normal(loc=0., scale=1e-3, size=M))
    print(step)
    try:
        les.step(actions, nIntermediate)
    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break
    print(les.gradient)
    step += 1
