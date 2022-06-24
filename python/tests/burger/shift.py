from numpy import pi
from scipy import interpolate
from scipy.fftpack import fft, ifft, fftfreq
import numpy as np


L = 2*pi

xdns = np.linspace(0, L, 1000)
x    = np.linspace(0, L, 32)


fdns = np.sin(xdns)

shift = 2.
x = x + shift
x[x>L] = x[x>L] - L
x[x<0] = x[x<0] + L

fsgs = np.sin(x)


shift = 0.
inter = interpolate.interp1d(xdns, fdns)
ftruth = inter(x)


mse = np.mean((fsgs - ftruth)**2)
print(mse)







