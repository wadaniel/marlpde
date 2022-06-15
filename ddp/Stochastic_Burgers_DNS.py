import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import pickle


L=100.0     # domainsize
nu=0.02     # viscosity 
A=np.sqrt(2)*1e-2 # scaling factor
N=512       # grid size / num Fourier modes
dt=0.01     # time step
s=20        # ratio of LES and DNS time steps
M=10000000  # number of timestes
P=1         # time steps between samples

# grid
x = np.linspace(0, L, N, endpoint=False)

# fourier modes
k = fftfreq(N, L / (2*np.pi*N))
k1  = 1j * k

u_old=np.sin(2.*np.pi*2.*x/L+np.random.normal()*2.*np.pi);

un_old=fft(u_old)
Fn_old=k1*fft(0.5*(u_old)**2)

# Storage for DNS field
U_DNS=np.zeros((N,M//P))
# Storage for forcing terms
f_store=np.zeros((N,M//P))

U_DNS[:,0]=u_old
z=0

u=u_old
un=np.zeros(N)

f=np.zeros(N)
for kk in range(1,4):
    C1=np.random.normal()
    C2=np.random.normal()
    f=f+C1*A/np.sqrt(kk*s*dt)*np.cos(2*np.pi*kk*x/L+2*np.pi*C2);

fn=fft(f)

# For integration step
C=0.5*k**2*nu*dt

for m in range(1,M):
    if (m % 1000 == 0):
        print(f"Step {m}")

    Fn=k1*fft(0.5*(u_old)**2)

    if(m%s==0):
        f = np.zeros(N);
        for kk in range(1,4):
            C1=np.random.normal()
            C2=np.random.normal()
            f=f+C1*A/np.sqrt(kk*s*dt)*np.cos(2*np.pi*kk*x/L+2*np.pi*C2)

        fn=fft(f)
    
    un=((1.0-C)*un_old-0.5*dt*(3.0*Fn-Fn_old)+dt*fn)/(1.0+C)
    
    un_old=un
    u=np.real(ifft(un))
    Fn_old=Fn
    
    if (m%P==0):
        f_store[:,z] = f
        z=z+1
        U_DNS[:,z] = u

f_store = f_store[:,0::s] # weird but ok

print(f"Storing U_DNS {U_DNS.shape}")
pickle.dump(U_DNS, open('/scratch/wadaniel/DNS_Burgers_s_20.pickle', 'wb'))
print(f"Storing f_store {f_store.shape}")
pickle.dump(f_store, open('/scratch/wadaniel/DNS_Force_LES_s_20.pickle', 'wb'))
