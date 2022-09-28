import os
import sys
from tqdm import trange

from Burger import Burger
import helpers
from helpers import swish
import numpy as np
import matplotlib.pyplot as plt

scratch = os.getenv("SCRATCH", default=".")
basedir = f'{scratch}/ddp'
if not os.path.exists(basedir):
    os.mkdir(basedir)

from scipy.fftpack import fftfreq
from scipy.stats import gaussian_kde

import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

M=int(1e6)
run=1


N=1024          # grid size / num Fourier modes
N_bar=128       # sgs grid size / num Fourier modes
nu=0.02         # viscosity 
noise=0.1       # noise for ic
nunoise=False
seed=42         # random seed
fseed=42        # random seed of forcing term
forcing=False   # apply forcing term during step
stepper=20      # ratio of LES and DNS time steps

L  = 2*np.pi    # domainsize
dt = 0.001      # time step
T  = 5          # terminal time
ic = "turbulence" # initial condition

#L  = 100       # domainsize
#dt = 0.01      # time step
#T  = 10000     # terminal time
#ic = "sinus"   # initial condition

checkpoint_path = f'{basedir}/best_model_weights_{ic}_{N}_{N_bar}_{run}.npz'

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


normalization = np.load(f'{basedir}/normalization_{ic}_{N}_{N_bar}_{run}.npz') 
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

full_input = np.load( f'{basedir}/u_bar_{ic}_{N}_{N_bar}_{run}.npy')
full_input = full_input.T

fbar_input = np.load( f'{basedir}/f_bar_{ic}_{N}_{N_bar}_{run}.npy')
fbar_input = fbar_input.T

full_output = np.load( f'{basedir}/PI_{ic}_{N}_{N_bar}_{run}.npy')
full_output = full_output.T

predict_input = full_input[train_region:train_region+predict_num,:]
predict_output = full_output[train_region:train_region+predict_num,:]

predict_input_normalized = (predict_input-normalization_mean)/normalization_std
predict_output_normalized = (predict_output-normalization_mean)/normalization_std

print(f'shape of input {predict_input.shape}')
print(f'shape of output {predict_output.shape}')


nu = 0.01
Lx  = 100
dx  = Lx/N_bar
x   = np.arange(N_bar)/L
kx  = fftfreq(N_bar, L / (2*np.pi*N_bar))


D1 = 1j*kx
D2 = kx*kx
D1 = D1.reshape([N_bar,1])
D2 = D2.reshape([N_bar,1])

D2x = 1 + 0.5*dt*nu*D2


u_store = np.zeros((N_bar, predict_num))
sub_store = np.zeros((N_bar, predict_num))

"""
u_old = full_input[pred_start-1,:].reshape([N_bar,1])
u = full_input[pred_start,:].reshape([N_bar,1])
u_fft = fft(u,axis=0)
u_old_fft = fft(u_old,axis=0)
subgrid_prev_n = model.predict(((u_old-mean_input)/std_input).reshape((1,N_bar))).reshape(N_bar,1)
subgrid_prev_n = subgrid_prev_n*std_output+mean_output

for i in range(maxit):
  subgrid_n = model.predict(((u-mean_input)/std_input).reshape((1,N_bar))).reshape(N_bar,1)
  subgrid_n = subgrid_n*std_output+mean_output

  force=force_bar[:,i].reshape((N_bar,1))

  F = D1*fft(.5*(u**2),axis=0)
  F0 = D1*fft(.5*(u_old**2),axis=0)
  
  uRHS = -0.5*dt*(3*F- F0) - 0.5*dt*nu*(D2*u_fft)  + u_fft + dt*fft(force,axis=0) \
              -fft(dt*3/2*subgrid_n + 1/2*dt*subgrid_prev_n,axis = 0)

  u_fft = uRHS/D2x.reshape([N_bar,1])
  u = np.real(ifft(u_fft,axis=0))
  u_store[:,i] = u.squeeze()
  sub_store[:,i] = subgrid_n.squeeze()

np.save(f'{basedir}/DDP_results_trained_{ic}_{N}_{N_bar}_{run}.npy', {'u_pred':u_store, 'sub_pred':sub_store})
"""


reps = 10
nT=int(T//dt)
kRelErr = np.zeros((reps*int(T//dt), N_bar//2))
actionHistory = np.zeros((reps*int(T//dt), N_bar))
u = np.zeros((reps*int(T//dt), N))
urg = np.zeros((reps*int(T//dt), N_bar))

for i in range(reps):
    print(f"Simulate DNS {i}..")
    dns = Burger(L=L, 
            N=N, 
            dt=dt, 
            nu=nu, 
            tend=T, 
            case=ic, 
            forcing=forcing, 
            noise=noise, 
            seed=seed+i,
            fseed=fseed+i,
            stepper=stepper,
            nunoise=nunoise)
    dns.simulate()
    dns.compute_Ek()
    u[i*int(T//dt):(i+1)*int(T//dt),:] = dns.uu[1:-1,:]
    print("DONE!")

    print(f"Simulate SGS {i}..")
    sgs = Burger(L=L, 
            N=N_bar, 
            dt=dt, 
            nu=nu, 
            tend=T, 
            case=ic, 
            forcing=forcing, 
            noise=noise, 
            seed=seed+i,
            fseed=fseed+i,
            stepper=stepper,
            nunoise=nunoise,
            sgsNN=model,
            u_shift=normalization_mean,
            u_scale=normalization_std)
    v0 = np.concatenate((dns.v0[:(N_bar+1)//2], dns.v0[-(N_bar-1)//2:])) * N_bar / N
    sgs.IC( v0 = v0 )

    sgs.simulate()
    sgs.compute_Ek()

    kRelErr[i*int(T//dt):(i+1)*int(T//dt),:] = (np.abs(dns.Ek_ktt[1:-1,:N_bar//2] - sgs.Ek_ktt[1:-1,:N_bar//2])/dns.Ek_ktt[1:-1,:N_bar//2])**2

    #kRelErr[i*int(T//dt):(i+1)*int(T//dt),:] /= (np.arange(1,nT+1)[:, np.newaxis]*dt)
    actionHistory[i*int(T//dt):(i+1)*int(T//dt),:] = sgs.actionHistory[1:-1,:]
    urg[i*int(T//dt):(i+1)*int(T//dt),:] = sgs.uu[1:-1,:]


np.save(f'{basedir}/relError_{ic}_{N}_{N_bar}_{run}.npy',kRelErr)
np.save(f'{basedir}/actionHistory_{ic}_{N}_{N_bar}_{run}.npy',kRelErr)
np.save(f'{basedir}/dns_{ic}_{N}_{N_bar}_{run}.npy', u)
np.save(f'{basedir}/urg_{ic}_{N}_{N_bar}_{run}.npy', urg)

print("DONE!")

kRelDiffErr = np.diff(kRelErr, axis=0)
print(kRelDiffErr)
kRelDiffErrMean = np.mean(kRelDiffErr, axis=0)
kRelDiffErrSdev = np.std(kRelDiffErr, axis=0)

print(kRelDiffErrMean)
print(kRelDiffErrSdev)

print(kRelErr.shape)
#avgRelError = kRelErr/(np.arange(1,nT+1)[:, np.newaxis]*dt)
#avgRelError = np.mean(avgRelError, axis=0)
avgRelError = kRelErr

k = np.arange(N_bar//2)
#qs = np.quantile(kRelDiffErr, q=[0.1, 0.5, 0.9], axis=0) 
qs = np.quantile(avgRelError, q=[0.1, 0.5, 0.9], axis=0) 
print(qs.shape)

## Plot relative error
plt.figure(0)
plt.plot(k, qs[1], color='coral')
plt.fill_between(k, y1=qs[0], y2=qs[2], alpha=0.2, color='coral')
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'rel_error_{ic}_{N}_{N_bar}_{run}.png')
plt.close()


## Plot density of SGS terms
fig, axs = plt.subplots(2, 1)

sgsTerms = actionHistory.flatten()
smax = sgsTerms.max()
smin = sgsTerms.min()
slevels = np.linspace(smin, smax, 50)
svals = np.linspace(smin, smax, 500)

sfac = 3
sgsMean = np.mean(sgsTerms)
sgsSdev = np.std(sgsTerms)
svals2  = np.linspace(sgsMean-sfac*sgsSdev,sgsMean+sfac*sgsSdev,500)
   
sgsDensity = gaussian_kde(sgsTerms)
axs[0].plot(svals, sgsDensity(svals), color='turquoise')
axs[1].plot(svals2, sgsDensity(svals2), color='turquoise')
plt.tight_layout()

plt.savefig(f'sgs_density_{ic}_{N}_{N_bar}_{run}.png')
plt.close()

## Plot SGS terms of one simulation
sgsField = actionHistory[:int(T//dt),:]

plt.figure()
plt.contourf(sgs.x, np.arange(nT)*dt, sgsField, cmap='plasma') #, slevels)
plt.tight_layout()
plt.savefig(f'sgs_field_{ic}_{N}_{N_bar}_{run}.png')
plt.close()
