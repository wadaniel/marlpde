import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import convolve
import py

N = 2048
L = 2*np.pi
h = L/N
T = 5
dt = 0.0001
x = np.arange(0, N)*h
D = 0.01
k  = np.r_[0:N/2, 0, -N/2+1:0]*2*np.pi/L # Wave numbers
k1 = 1j*k
k2 = k1**2

uinit = np.sin(x)
u = uinit
v = fft(u)

uu = np.zeros((N, int(T/dt)+1))
uu[:, 0] = u


for t in range(int(T/dt)):
    v = v - dt*k1*0.5*fft(u**2) + dt*D*k2*v
    #v = v - dt*k1*0.5*convolve(v,v, 'same') + dt*D*k2*v ## SLOW
    u = np.real(ifft(v))
    uu[:, t+1] = u



fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
for i in range(16):
    t = int(i * T / dt / 16)
    k = int(i / 4)
    l = i % 4
    axs[k,l].plot(x, uu[:,t])

fig.savefig('simple.png'.format())
plt.close()
    


