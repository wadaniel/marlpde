import os
import sys
from tqdm import trange

#import h5py
import helpers
from helpers import swish
import numpy as np
import matplotlib.pyplot as plt

scratch = os.getenv("SCRATCH", default=".")
basedir = f'{scratch}/ddp'
if not os.path.exists(basedir):
    os.mkdir(basedir)

from scipy.fftpack import fftfreq

import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

M=int(1e6)

N=1024        # grid size / num Fourier modes
N_bar=128      # sgs grid size / num Fourier modes
nu=0.02       # viscosity 
noise=0.1     # noise for ic
seed=42       # random seed
forcing=True  # apply forcing term during step
s=20          # ratio of LES and DNS time steps

#L  = 2*np.pi   # domainsize
#dt = 0.001      # time step
#T  = 100         # terminal time
#ic = "turbulence" # initial condition

L  = 100       # domainsize
dt = 0.01      # time step
T  = 10000     # terminal time
ic = "sinus"   # initial condition

checkpoint_path = f'{basedir}/best_model_weights.npz'

model = Sequential()
model.add(Dense(N_bar,input_shape=(N_bar,),activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(N_bar,activation=None))
model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.load_weights(checkpoint_path)


normalization = np.load(f'{basedir}/normalization.npz') #, allow_pickle=True)
normalization_mean = normalization['mean_input']
normalization_std = normalization['std_input']
print(f'normalization mean : {normalization_mean}')
print(f'normalization std : {normalization_std}')

train_region = 1000000
train_num = 500000
predict_num = 500000
assert train_num <= train_region

x = np.arange(N)/L
# Storage for DNS field
U_DNS=np.zeros((N,M), dtype=np.float16)
# Storage for forcing terms
f_store=np.zeros((N,M), dtype=np.float16)

full_input = np.load( f'{basedir}/u_bar.npy')
full_input = full_input.T

fbar_input = np.load( f'{basedir}/f_bar.npy')
fbar_input = fbar_input.T

full_output = np.load( f'{basedir}/PI.npy')
full_output = full_output.T

predict_input = full_input[train_region:train_region+predict_num,:]
predict_output = full_output[train_region:train_region+predict_num,:]

predict_input_normalized = (predict_input-normalization_mean)/normalization_stdev
predict_output_normalized = (predict_output-normalization_mean)/normalization_stdev

print(f'shape of input {predict_input.shape}')
print(f'shape of output {predict_output.shape}')


s=20

NX = 128
nu = 0.01

dt = s*1e-2

Lx  = 100
dx  = Lx/NX
x   = np.arange(N)/L
kx  = fftfreq(N, L / (2*np.pi*N))


D1 = 1j*kx
D2 = kx*kx
D1 = D1.reshape([NX,1])
D2 = D2.reshape([NX,1])

D2x = 1 + 0.5*dt*nu*D2


u_store = np.zeros((NX, predict_num))
sub_store = np.zeros((NX, predict_num))

force_dict = pickle.load( open(f'{basedir}/f_bar.npy', 'rb') )
force_bar=force_dict[:,train_region:train_region+num_pred]

sys.exit()

u_old = full_input[pred_start-1,:].reshape([NX,1])
u = full_input[pred_start,:].reshape([NX,1])
u_fft = fft(u,axis=0)
u_old_fft = fft(u_old,axis=0)
subgrid_prev_n = model.predict(((u_old-mean_input)/std_input).reshape((1,128))).reshape(128,1)
subgrid_prev_n = subgrid_prev_n*std_output+mean_output

for i in range(maxit):
  subgrid_n = model.predict(((u-mean_input)/std_input).reshape((1,128))).reshape(128,1)
  subgrid_n = subgrid_n*std_output+mean_output

  force=force_bar[:,i].reshape((NX,1))

  F = D1*fft(.5*(u**2),axis=0)
  F0 = D1*fft(.5*(u_old**2),axis=0)
  
  uRHS = -0.5*dt*(3*F- F0) - 0.5*dt*nu*(D2*u_fft)  + u_fft + dt*fft(force,axis=0) \
              -fft(dt*3/2*subgrid_n + 1/2*dt*subgrid_prev_n,axis = 0)

  ## RK3
  #v = u_fft
  #fn = fft(force,axis=0)

  #v1 = v + dt * (-0.5*D1*fft(u**2) - nu*D2*v + fn)
  #u1 = np.real(ifft(v1))

  #v2 = 3./4.*v + 1./4.*v1 + 1./4. * dt * (-0.5*D1*fft(u1**2) - nu*D2*v1 + fn)
  #u2 = np.real(ifft(v2))

  #v3 = 1./3.*v + 2./3.*v2 + 2./3. * dt * (-0.5*D1*fft(u2**2) - nu*D2*v2 + fn)
  #u_fft = v3

  #subgrid_prev_n = subgrid_n
  #u_old_fft = u_fft
  #u_old = u

  u_fft = uRHS/D2x.reshape([NX,1])
  u = np.real(ifft(u_fft,axis=0))
  u_store[:,i] = u.squeeze()
  sub_store[:,i] = subgrid_n.squeeze()

np.save(f'{basedir}/DDP_results_trained_{int(train_num/1000)}_region_{region}_new.npy', {'u_pred':u_store, 'sub_pred':sub_store})
