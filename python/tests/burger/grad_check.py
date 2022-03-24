#!/bin/python3

""" Script to test absis function """

import sys
sys.path.append('./../../_model/')

from Burger_jax2 import *
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, random

#!/bin/python3

""" Script to test absis function """

from Burger_jax2 import *

N    = 64
L    = 2*np.pi
dt   = 0.0005
tEnd = 5
nu   = 0.01
gridSize = 64
episodeLength = 500

# action defaults
basis = 'hat'
numActions = 16

# Initialize LES
les = Burger_jax2(L=L, N=gridSize, dt=dt, nu=nu, tend=tEnd)
les.setup_basis(numActions, basis)

## run controlled simulation
error = 0
step = 0
nIntermediate = int(tEnd / dt / episodeLength)
cumreward = 0.
while step < episodeLength and error == 0:

    # apply action and advance environment
    actions = jnp.asarray(np.random.normal(loc=0., scale=1e-4, size=numActions))
    print(step)
    try:
        for _ in range(nIntermediate):

            les.step(actions)
        les.compute_Ek()
    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break
    step += 1
print(les.gradient)
