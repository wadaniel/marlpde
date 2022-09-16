import os
import sys
from tqdm import trange

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

N = 1024
N_bar=128      # sgs grid size / num Fourier modes

checkpoint_path = f'{basedir}/best_model_weights_{N}_{N_bar}.npz'

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

full_input = np.load( f'{basedir}/u_bar_{N}_{N_bar}.npy')
full_input = full_input.T

full_output = np.load( f'{basedir}/PI_{N}_{N_bar}.npy')
full_output = full_output.T

predict_input_normalized = (full_input-normalization_mean)/normalization_std

print(f'shape of input {predict_input_normalized.shape}')

predict = np.zeros(full_output.shape)
for i in trange(len(predict_input_normalized)):
    predict[i,:] = (model.predict(predict_input_normalized[i,:].reshape((1,N_bar))).reshape((N_bar,)) + normalization_mean)*normalization_std

np.save(f'{basedir}/PI_predict_prior.npy', predict)
print('DONE!')
