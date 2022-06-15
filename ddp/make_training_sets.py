import os
import pickle
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq

# Box filter in Fourier space, 'u' field to be filtered, 'N' length of 'u', 'n_sub' length of output, number of filtered Fourier modes
def filter_bar(u, N, n_sub):
    v = fft(u)
    vbar = np.concatenate((v[:((n_sub+1)//2)], v[-(n_sub-1)//2:]))
    ubar = np.real(ifft(vbar)) * n_sub / N

    return ubar

# Calculates filtered field 'u_bar' and filtered forcing term 'f_bar', and SG term PI
def calc_bar(U_DNS, f_store, NX, NY):
    Lx=100

    f_bar = filter_bar(f_store,NX,NY)
    u_bar = filter_bar(U_DNS,NX,NY)

    U2_DNS = U_DNS**2
    u2_bar = filter_bar(U2_DNS,NX,NY)
    
    tau = .5*(u2_bar - u_bar**2)
    mtau = np.roll(tau, 1)
    
    PI = (tau-mtau)/(Lx/NY)

    return (u_bar, PI, f_bar)

scratch = os.getenv("SCRATCH", default=".")
N_bar= 128              # gridpoints / fourier modes of filtered field
s = 20                  # ratio of LES and DNS time steps
num_in_set = 1250000    # size of dataset

assert num_in_set % s == 0

print("Loading data..")
U_DNS = pickle.load( open('{}/DNS_Burgers_s_20.pickle'.format(scratch), 'rb') )
f_store = pickle.load( open('{}/DNS_Force_LES_s_20.pickle'.format(scratch), 'rb') )

print(U_DNS.shape)
print(f_store.shape)

# shift between start of datasets
set_size = 250000

print("Compute filtered data..")
for i in range(13,14):

    u = U_DNS[:,(i-1)*set_size:((i-1)*set_size)+num_in_set]
    f = f_store[:,(i-1)*(set_size//s):((i-1)*(set_size//s)+(num_in_set//s))]
    
    u_bar, PI, f_bar = calc_bar(u, f, 1024, N_bar)
    
    print(u_bar.shape)
    print(f_bar.shape)
    print(PI.shape)
    
    np.save('{}/u_bar_region_{}.npy'.format(scratch, i),u_bar)
    np.save('{}/f_bar_region_{}.npy'.format(scratch, i),f_bar)
    np.save('{}/PI_region_{}.npy'.format(scratch, i),PI)
