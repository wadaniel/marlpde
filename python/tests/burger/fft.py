import numpy as np

N = 16
gridSize = 8

#f = np.fft.fftfreq(N, 1. / N)
print(f)

x = np.linspace(0, 2*np.pi, N)
print("x")
print(x)

u = np.sin(x)
#u = np.zeros(N)
#u[0]=1
print("u")
print(u)

v = np.fft.fft(u)
print("v")
print(v)


iu = np.real(np.fft.ifft(v))
print("iu")
print(iu)

sampleSpacing=1
freq = np.fft.fftfreq(N, d=sampleSpacing)

print("freq")
print(freq)

w = np.zeros(N, dtype='complex')
w[abs(freq)<1./4] = v[abs(freq)<1./4.]
#w=v
print("w")
print(w)

it = np.real(np.fft.ifft(w))
print("it")
print(it)




