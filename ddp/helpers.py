import numpy as np
from scipy.fftpack import fft, ifft

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
 
    dx = Lx/NY
    PI = (tau-mtau)/dx

    return (u_bar, PI, f_bar)


