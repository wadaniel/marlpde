import numpy as np
from tensorflow import keras
from scipy.fftpack import fft, ifft

# Box filter in Fourier space, 'u' field to be filtered, 'N' length of 'u', 'n_sub' length of output, number of filtered Fourier modes
def filter_bar(u, N, n_sub):
    v = fft(u)
    vbar = np.concatenate((v[:((n_sub+1)//2)], v[-(n_sub-1)//2:]))
    ubar = np.real(ifft(vbar)) * n_sub / N

    return ubar


# Calculates filtered field 'u_bar' and filtered forcing term 'f_bar', and SG term PI
def calc_bar(U_DNS, f_store, NX, NY, Lx=100):

    f_bar = filter_bar(f_store,NX,NY)
    u_bar = filter_bar(U_DNS,NX,NY)

    U2_DNS = U_DNS**2
    u2_bar = filter_bar(U2_DNS,NX,NY)
    
    tau = .5*(u2_bar - u_bar**2)
    mtau = np.roll(tau, 1)
 
    dx = Lx/NY
    PI = (tau-mtau)/dx

    return (u_bar, PI, f_bar)

def normalize_data(data):

  std_data = np.std(data)
  mean_data = np.mean(data)

  norm_data = (data-mean_data)/std_data

  return norm_data, mean_data, std_data

def swish(x):
   beta = 1.0
   return beta * x * keras.backend.sigmoid(x)

def shift_data(data1,data2):
  shifts = np.random.randint(0,data1.shape[1],data1.shape[0])
  for i in range(data1.shape[0]):
    data1[i,:] = np.concatenate((data1[i,shifts[i]:], data1[i,:shifts[i]]))
    data2[i,:] = np.concatenate((data2[i,shifts[i]:], data2[i,:shifts[i]]))

  return data1, data2


