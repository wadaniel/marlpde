import os
import pickle
from helpers import *
import numpy as np
import scipy
import scipy.sparse as sparse
from scipy.sparse import linalg
import math
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow.keras.layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from scipy.io import savemat
from scipy.io import loadmat
from scipy.fftpack import fft, ifft
from helpers import *

train_num = 500000
region = "13"

train_region = 1000000
train_start = 0
num_pred = 20000


scratch     = os.getenv("SCRATCH", default=".")
u_bar_dict  = np.load(f'{scratch}/u_bar_region_{region}.npy')
full_input=u_bar_store=u_bar_dict.T

full_output = np.load('{scratch}/PI_region_{region}.npy')
full_output = full_output.T

full_input[:train_region,:], full_output[:train_region,:] = shift_data(full_input[:train_region,:], full_output[:train_region,:])

norm_input, mean_input, std_input = normalize_data(full_input[:train_region,:])

norm_output, mean_output, std_output = normalize_data(full_output[:train_region,:])

training_input = norm_input
training_output = norm_output

print('shape of input')
print(np.shape(training_input))

print('shape of output')
print(np.shape(training_output))

index=np.random.permutation(train_region)

print(std_input)
print(std_output)

print(mean_input)
print(mean_output)

input_train=training_input[index[0:train_num],:]
output_train=training_output[index[0:train_num],:]

test_input=training_input[index[train_num:(train_num+num_pred)],:]
test_output=training_output[index[train_num:(train_num+num_pred)],:]


model = Sequential()

model.add(Dense(128,input_shape=(128,),activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(128,activation=None))

model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(input_train, output_train,epochs=100,batch_size=200,shuffle=True,validation_split=0.2)

model.save_weights('./weights_trained_ANN')
exit()

pred_start = train_region + 50000

s=20

NX = 128
nu = 2e-2

dt = s*1e-2

Lx    = 100
dx    = Lx/NX
x     = np.linspace(0, Lx, num=NX)
#kx    = (2*math.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=float),np.arange((-NX/2+1),0,dtype=float))).reshape([NX,1])
kx     = fftfreq(N, L / (2*np.pi*N))


maxit=100000

D1 = 1j*kx
D2 = kx*kx
D1 = D1.reshape([NX,1])
D2 = D2.reshape([NX,1])
D2_tensor = np.float32((D2[0:int(NX/2)]-np.mean(D2[0:int(NX/2)])/np.std(D2[0:int(NX/2)])))

D2x = 1 + 0.5*dt*nu*D2


u_store = np.zeros((NX,maxit))
sub_store = np.zeros((NX,maxit))

reg = 13

force_dict = pickle.load( open(f'{scratch}/f_bar_all_regions.pickle', 'rb') )
force_bar=force_dict[:,int((reg-1)*12500)+int(pred_start/s):]

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

np.save(f'{scratch}/DDP_results_trained_{int(train_num/1000)}_region_{region}_new.npy', {'u_pred':u_store, 'sub_pred':sub_store})
