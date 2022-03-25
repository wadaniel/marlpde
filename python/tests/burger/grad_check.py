#!/bin/python3

""" Script to test absis function """

import sys
sys.path.append('./../../_model/')

from Burger_jax2 import *
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, random
from numpy.linalg import norm

#!/bin/python3

""" Script to test absis function """

from Burger_jax_rep import *
from Burger_jax2 import *

N    = 64
L    = 2*np.pi
dt   = 0.0005
tEnd = 5
nu   = 0.01
episodeLength = 500

# action defaults
basis = 'hat'
M = 16
eps = 1e-8
ind = 6 #infinitesimal modification of action i = ind
grad_a = np.zeros(N)

# Initialize LES
les = Burger_jax_rep(L=L, N=N, dt=dt, nu=nu, tend=tEnd)
les_a = Burger_jax_rep(L=L, N=N, dt=dt, nu=nu, tend=tEnd)
les.setup_basis(M, basis)
les_a.setup_basis(M, basis)
## run controlled simulation
error = 0
step = 0
#nIntermediate = int(tEnd / dt / episodeLength)
nIntermediate = 10
cumreward = 0.
while step < episodeLength and error == 0:

    # apply action and advance environment
    actions = np.random.normal(loc=0., scale=1e-4, size=M)
    #actions = np.ones(M)
    actions_a = actions.copy()
    actions_a[ind] += eps
    print(step)
    try:
        #for _ in range(nIntermediate):
        les.step(actions, nIntermediate)
        les_a.step(actions_a, nIntermediate)
        grad_a = (les_a.u - les.u)/eps
        print(grad_a)
        grad_ind = les.gradient[:,ind]
        print(grad_ind)
        err = norm(grad_a - grad_ind)/(norm(grad_a) + norm(grad_ind))
        print(err)
        les_a.v = les.v.copy()
        les_a.u = les.u.copy()
    except Exception as e:
        print("Exception occured:")
        print(str(e))
        error = 1
        break
    step += 1
