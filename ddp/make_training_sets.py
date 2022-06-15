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

print("Loading data..")
U_DNS = pickle.load( open('/scratch/wadaniel/DNS_Burgers_s_20.pickle', 'rb') )
f_store = pickle.load( open('/scratch/wadaniel/DNS_Force_LES_s_20.pickle', 'rb') )

print(U_DNS.shape)
print(f_store.shape)

s=20                    # ratio of LES and DNS time steps
num_in_set = 1250000    # size of dataset

assert num_in_set % s == 0

# shift between start of datasets
set_size = 250000

print("Compute filtered data..")
for i in range(12,13):

    u = U_DNS[:,(i-1)*set_size+1:((i-1)*set_size)+num_in_set+1]
    print(u.shape)
    f = f_store[:,(i-1)*(set_size//s)+1:((i-1)*(set_size//s)+(num_in_set//s))]
    print(f.shape)
    
    u_bar, PI, f_bar = calc_bar(u, f, 1024, 128)
    
    np.save('/scratch/wadaniel/u_bar_region_{}.npz'.format(i),u_bar)
    np.save('/scratch/wadaniel/f_bar_region_{}.npz'.format(i),f_bar)
    np.save('/scratch/wadaniel/PI_region_{}.npz'.format(i),PI)
end
