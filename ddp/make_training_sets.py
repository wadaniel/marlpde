import os
import pickle
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from helpers import *

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
    
    print(u.shape)
    print(u[:, 3])
    print(u_bar.shape)
    print(u_bar[:, 3])
    print(PI.shape)
    print(PI[:, 3])
    print(f_bar.shape)
    print(f_bar[:, 3])
    
    np.save('{}/u_bar_region_{}.npy'.format(scratch, i),u_bar)
    np.save('{}/f_bar_region_{}.npy'.format(scratch, i),f_bar)
    np.save('{}/PI_region_{}.npy'.format(scratch, i),PI)
