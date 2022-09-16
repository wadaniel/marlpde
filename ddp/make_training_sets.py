import os
import pickle
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from helpers import *

scratch = os.getenv("SCRATCH", default=".")
basedir = f'{scratch}/ddp'
N_bar = 128              # gridpoints / fourier modes of filtered field
s = 20                  # ratio of LES and DNS time steps
num_in_set = 1250000    # size of dataset

M = int(1e6)
N = 1024

assert num_in_set % s == 0

print("Loading data..")
with open(f'{basedir}/DNS_Burgers_s{s}_M{M}_N{N}.pickle', 'rb') as f:
    U_DNS = pickle.load( f )
with open(f'{basedir}/DNS_Force_LES_s{s}_M{M}_N{N}.pickle', 'rb') as f:
    f_store = pickle.load( f )

print(f'U_DNS {U_DNS.shape}')
print(f'f_store {f_store.shape}')

# shift between start of datasets
shift_set = 250000

print("Compute filtered data..")
for i in range(0,2):

    u = U_DNS[:,i*shift_set:(i*shift_set)+num_in_set]
    f = f_store[:,i*(shift_set//s):(i*(shift_set//s)+(num_in_set//s))]
    
    print(f'u shape {u.shape}')
    print(f'f shape {f.shape}')
    u_bar, PI, f_bar = calc_bar(u, f, 1024, N_bar)
    
    np.save(f'{basedir}/u_bar_region_{i}_set_{num_in_set}.npy',u_bar)
    np.save(f'{basedir}/f_bar_region_{i}_set_{num_in_set}.npy',f_bar)
    np.save(f'{basedir}/PI_region_{i}_set_{num_in_set}.npy',PI)
