import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from numpy.linalg import norm

# def sum_logistic(x):
#       return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))
#
# x_small = jnp.arange(3.)
# derivative_fn = grad(sum_logistic)
# print(derivative_fn(x_small))
x = [2, 2, 2]
norm = norm(x)
print(norm)
