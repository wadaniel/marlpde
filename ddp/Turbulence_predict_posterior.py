import os
import sys
from tqdm import trange
sys.path.append('../python/_model')
from Burger import Burger

import helpers
from helpers import swish
import numpy as np
import matplotlib.pyplot as plt

scratch = os.getenv("SCRATCH", default=".")
basedir = f'{scratch}/ddp'

from scipy.fftpack import fftfreq

import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

N_bar=32      # sgs grid size / num Fourier modes
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
T  = 1000      # terminal time
ic = "sinus"   # initial condition

nunoise = False

episodelength = int(T/dt)

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

sgs = Burger(L=L, 
        N=N_bar, 
        dt=dt, 
        nu=nu, 
        tend=T, 
        case=ic, 
        forcing=forcing, 
        noise=0., 
        seed=seed, 
        s=s,
        version=0, 
        nunoise=nunoise, 
        numAgents = 1)

sgs.setup_basis(N_bar)

pi_predict = np.zeros((episodelength,N_bar))

for i in trange(episodelength):
    u_normalized = (sgs.u-normalization_mean)/normalization_std
    pi_predict[i,:] = (model.predict(u_normalized.reshape((1,N_bar))).reshape((N_bar,)) + normalization_mean)*normalization_std
    sgs.step(pi_predict[i,:])


np.save(f'{basedir}/PI_predict_posterior.npy', pi_predict)


