import pickle
import numpy as np

def filter_bar(u, N, n_sub):
    var_bar = np.zeros((N/n_sub,length(u)))
    for i in range(N//n_sub):
        var_bar[i,:]=mean(u[n_sub*(i-1)+1:n_sub*(i-1)+n_sub,:])

    return var_bar

# Calculates filtered field 'u_bar' and filtered forcing term 'f_bar', and SG term PI
def calc_bar(U_DNS, f_store, NX, NY):
    Lx=100

    kx_bar = fftfreq(NY, L / (2*np.pi*NY))

    full_f = f_store

    f_bar = filter_bar(full_f,NX,NX/NY)

    full_u = U_DNS

    u_bar = filter_bar(full_u,NX,NX/NY)

    uu_full = full_u**2

    uu_coarse = filter_bar(uu_full,NX,NX/NY)

    tau = .5*(uu_coarse - u_bar**2)

    k1  = 1j * kx
    fft_PI = k1*fft(tau)

    PI = real(ifft(fft_PI))

    return (u_bar, PI, f_bar)

U_DNS = pickle.load( open('/scratch/wadaniel/DNS_Burgers_s_20.pickle', 'rb') )
f_store = pickle.load( open('/scratch/wadaniel/DNS_Force_LES_s_20.pickle', 'rb') )

s=20                    # ratio of LES and DNS time steps
num_in_set = 1250000    # size of dataset

# shift between start of datasets
set_size = 250000

for i in range(12,13):

    u = U_DNS[:,(i-1)*set_size+1:((i-1)*set_size)+num_in_set]
    f = f_store[:,(i-1)*(set_size/s)+1:((i-1)*(set_size/s)+(num_in_set/s))]
    
    u_bar, PI, f_bar = calc_bar(u, f, 1024, 128)
    
    np.save('/scratch/wadaniel/u_bar_region_{}.npz'.format(i),u_bar)
    np.save('/scratch/wadaniel/f_bar_region_{}.npz'.format(i),f_bar)
    np.save('/scratch/wadaniel/PI_region_{}.npz'.format(i),PI)
end
