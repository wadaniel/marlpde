import numpy as np
from scipy.fftpack import fft, ifft, fftfreq

import matplotlib.pyplot as plt

np.random.seed(123)

L = 2*np.pi
N = 1024
N_bar = 512

k = fftfreq(N, d=1/N) #, self.L / (2*np.pi*self.N))

x = np.linspace(0, L, N)
u = np.sin(x)
v = fft(u)

vbar = v.copy()
vbar[abs(k)>N_bar/2] = 0

x2 = np.linspace(0, L, N_bar)
vbar2 = np.concatenate((v[:((N_bar+1)//2)], v[-(N_bar-1)//2:]))

ui = np.real(ifft(v))
uibar = np.real(ifft(vbar))

uibar = uibar[0::N//N_bar]
uibar2 = np.real(ifft(vbar2)) * N_bar / N

plt.plot(x,u, 'k')
plt.plot(x,ui, 'k-')
plt.plot(x2,uibar)
plt.plot(x2,uibar2)
plt.savefig('filter_test.pdf')

kx_bar = fftfreq(N_bar, 1/N_bar)
#print(kx_bar)
fft_du = 1j*kx_bar*vbar2
du = np.real(ifft(fft_du)) * N_bar / N


um = np.roll(uibar, 1)
dx = L/N_bar
dudx = (uibar-um)/dx

#print(du)
#print(dudx)


print(np.mean((uibar-uibar2)**2))
print(np.mean((du-dudx)**2))
print(np.mean((du-np.cos(x2))**2))
print(np.mean((dudx-np.cos(x2))**2))



