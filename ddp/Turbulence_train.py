import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import pickle
import helpers
import numpy as np

from tqdm import trange
from helpers import swish

scratch = os.getenv("SCRATCH", default=".")
basedir = f'{scratch}/ddp'
if not os.path.exists(basedir):
    os.mkdir(basedir)

import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

M=int(1e6)

N=1024          # grid size / num Fourier modes
N_bar=32        # sgs grid size / num Fourier modes
ic = "turbulence" # initial condition

#N=1024         # grid size / num Fourier modes
#N_bar=128       # sgs grid size / num Fourier modes
#ic = "sinus"   # initial condition

run_load=1
run=1
nunoise=False

epochs = 1000 # number of epochs to train

train_num = 500000 #run 1
train_region = M
assert train_num <= train_region

# Storage for DNS field
U_DNS=np.zeros((N,M), dtype=np.float16)
# Storage for forcing terms
f_store=np.zeros((N,M), dtype=np.float16)

full_input = np.load( f'{basedir}/u_bar_{ic}_{N}_{N_bar}_{run_load}.npy')
full_input = full_input.T

full_output = np.load( f'{basedir}/PI_{ic}_{N}_{N_bar}_{run_load}.npy')
full_output = full_output.T

# Randomly shift data left/right
full_input[:train_region,:], full_output[:train_region,:] = helpers.shift_data(full_input[:train_region,:], full_output[:train_region,:])

norm_input, mean_input, std_input = helpers.normalize_data(full_input[:train_region,:])
norm_output, mean_output, std_output = helpers.normalize_data(full_output[:train_region,:])

np.savez(f'{basedir}/normalization_{ic}_{N}_{N_bar}_{run}.npz', mean_input=mean_input, std_input=std_input)

# Shuffle ordering
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

filepath = f'best_model_weights_{ic}_{N}_{N_bar}_{run}.npz'
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

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=opt, metrics=['mae'])
history = model.fit(training_input, training_output, epochs=epochs, batch_size=200, shuffle=True, validation_split=0.2, callbacks=callbacks)

pickle_file = open(f'{basedir}/history_ANN_{ic}_{epochs}_{N}_{N_bar}_{run}.pickle', 'wb')
pickle.dump(history.history,pickle_file)
pickle_file.close()

model.save_weights(f'{basedir}/weights_trained_ANN_{ic}_{epochs}_{N}_{N_bar}_{run}')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
fig, axs = plt.subplots(2, 1, figsize=(10,10)) 
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].set_ylabel('loss')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'test'], loc='upper left')

# summarize history for loss
axs[1].plot(history.history['mae'])
axs[1].plot(history.history['val_mae'])
axs[1].set_ylabel('mean average error')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'test'], loc='upper left')
plt.savefig(f'TF_training_{ic}_{epochs}_{N}_{N_bar}_{run}.png')
