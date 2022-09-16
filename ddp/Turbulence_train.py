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


import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

M=int(1e6)

N=1024        # grid size / num Fourier modes
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
T  = 10000     # terminal time
ic = "sinus"   # initial condition

nunoise=False

train_region = 1000000
train_num = 1000000
assert train_num <= train_region

# domain discretization
x = np.arange(N)/L
# Storage for DNS field
U_DNS=np.zeros((N,M), dtype=np.float16)
# Storage for forcing terms
f_store=np.zeros((N,M), dtype=np.float16)

full_input = np.load( f'{basedir}/u_bar.npy')
full_input = full_input.T

full_output = np.load( f'{basedir}/PI.npy')
full_output = full_output.T


full_input[:train_region,:], full_output[:train_region,:] = helpers.shift_data(full_input[:train_region,:], full_output[:train_region,:])

norm_input, mean_input, std_input = helpers.normalize_data(full_input[:train_region,:])
norm_output, mean_output, std_output = helpers.normalize_data(full_output[:train_region,:])

np.savez(f'{basedir}/normalization.npz', mean_input=mean_input, std_input=std_input)

random_index = np.random.permutation(train_region)
train_index = random_index[:train_num]

training_input = norm_input[train_index,:]
training_output = norm_output[train_index,:]

print(f'shape of input {training_input.shape}')
print(f'shape of output {training_output.shape}')

print(f'std_input: {std_input}')
print(f'std_output: {std_output}')

print(f'mean_input: {mean_input}')
print(f'mean_output: {mean_output}')

#filepath = 'best_model_weights.epoch{epoch:02d}-loss{val_loss:.2f}.npz'
filepath = 'best_model_weights.npz'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'{basedir}/{filepath}',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

callbacks = [model_checkpoint_callback]

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
#model.fit(training_input, training_output, epochs=20, batch_size=200, shuffle=True, validation_split=0.2, callbacks=callbacks)
model.fit(training_input, training_output, epochs=200, batch_size=200, shuffle=True, validation_split=0.2, callbacks=callbacks)

model.save_weights(f'{basedir}/weights_trained_ANN')
