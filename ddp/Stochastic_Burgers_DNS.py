import os
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import pickle

np.random.seed(1234)
scratch = os.getenv("SCRATCH", default=".")

L=100.0     # domainsize
nu=0.02     # viscosity 
A=np.sqrt(2)*1e-2 # scaling factor for forcing
N=1024      # grid size / num Fourier modes
dt=0.01     # time step
s=20        # ratio of LES and DNS time steps
M=int(1e5)  # number of timesteps
P=1         # time steps between samples
out=False # plot files
dump=True # dump fields

# grid
x = np.linspace(0, L, N, endpoint=False)

# fourier modes
#ka = np.arange(0, N/2 + 0.5, 1)
#kb = np.arange(-N/2+1, -0.5, 1)
#k = np.concatenate([ka, kb])*(2*np.pi/L)
k = fftfreq(N, L / (2*np.pi*N))
k1  = 1j * k
k2  = -k**2

u=np.sin(2.*np.pi*2.*x/L+np.random.normal()*2.*np.pi)
v=fft(u)
Fn_old=k1*fft(0.5*u**2)

# Storage for DNS field
U_DNS=np.zeros((N,M//P))
# Storage for forcing terms
f_store=np.zeros((N,M//P))

U_DNS[:,0]=u
z=0

fn=np.zeros(N)
for m in range(M):

    if (m % 1000 == 0):
        print(f"Step {m}")

    if(m%s==0):
        f = np.zeros(N)
        for kk in range(1,4):
            r1=np.random.normal()
            r2=np.random.normal()
            f += r1*A/np.sqrt(kk*s*dt)*np.cos(2*np.pi*kk*x/L+2*np.pi*r2)

        fn=fft(f)
    
    ## Adam Bashfort + CN
    C=-0.5*k2*nu*dt
    Fn=k1*fft(0.5*u**2)
    v=((1.0-C)*v-0.5*dt*(3.0*Fn-Fn_old)+dt*fn)/(1.0+C)
    Fn_old = Fn.copy()
    
    ## RK3
    """
    v1 = v + dt * (-0.5*k1*fft(u**2) + nu*k2*v + fn)
    u1 = np.real(ifft(v1))
        
    v2 = 3./4.*v + 1./4.*v1 + 1./4. * dt * (-0.5*k1*fft(u1**2) + nu*k2*v1 + fn)
    u2 = np.real(ifft(v2))

    v3 = 1./3.*v + 2./3.*v2 + 2./3. * dt * (-0.5*k1*fft(u2**2) + nu*k2*v2 + fn)
    v = v3
    """

    u=np.real(ifft(v))
    
    if (m%P==0):
        f_store[:,z] = f
        U_DNS[:,z] = u
        z=z+1

    if (out == True and ((m % (M//10)) == 0)):
        plt.plot(x,u)
        fname = f"u_field_{m}_{M}.pdf"
        print("Plotting " + fname)
        plt.savefig(fname)
        plt.close()

if dump:
    f_store = f_store[:,0::s]

    print(f"Storing U_DNS {U_DNS.shape}")
    with open(f'{scratch}/DNS_Burgers_s{s}_M{M}_N{N}.pickle', 'wb') as f:
        pickle.dump(U_DNS, f)

    print(f"Storing f_store {f_store.shape}")
    with open(f'{scratch}/DNS_Force_LES_s{s}_M{M}_N{N}.pickle', 'wb') as f:
        pickle.dump(f_store, f)
