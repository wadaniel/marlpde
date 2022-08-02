import os
import sys
from tqdm import trange
sys.path.append('../python/_model')
from Burger import Burger

import pickle
import helpers
from helpers import swish
import numpy as np
import matplotlib.pyplot as plt

scratch = os.getenv("SCRATCH", default=".")

import tensorflow.keras.layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

M=int(1e5)

#L=2*np.pi   # domainsize
L=100   # domainsize
N=1024      # grid size / num Fourier modes
N_bar=128   # sgs grid size / num Fourier modes
#dt=0.001    # time step
dt=0.01    # time step
nu=0.02     # viscosity 
#T=5         # terminal time
T=1000       # terminal time
#ic="turbulence" # initial condition
#ic="forced" # initical condition
ic="sinus" # initical condition
#forcing=False # apply forcing term during step
forcing=True  # apply forcing term during step
noise=0.1   # noise for ic
seed=42     # random seed
#s=1         # ratio of LES and DNS time steps
s=20         # ratio of LES and DNS time steps

nunoise=False

plot=True   # create plot
dump=True   # dump fields
load=False  # load fields

# domain discretization
x = np.arange(N)/L
# Storage for DNS field
U_DNS=np.zeros((N,M))
# Storage for forcing terms
f_store=np.zeros((N,M))

if load == False:
    ns = int(T/dt)
    nm = int(M/ns)
    print(f'num steps {ns}, num simulations {nm}')
    for i in trange(nm):
        dns = Burger(L=L, 
                N=N, 
                dt=dt, 
                nu=nu, 
                tend=T, 
                case=ic, 
                forcing=forcing, 
                noise=noise, 
                seed=seed, 
                s=s,
                version=0, 
                nunoise=nunoise, 
                numAgents=1)

        dns.simulate()
        
        U_DNS[:,i*ns:(i+1)*ns] = np.transpose(dns.uu[:ns,:])
        f_store[:,i*ns:(i+1)*ns] = np.transpose(dns.f[:ns,:])

    u_bar, PI, f_bar = helpers.calc_bar(U_DNS, f_store, N, N_bar, L)


    if (plot == True):
        figName = "evolution.pdf"
        print("Plotting {} ...".format(figName))
          
        fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
        for i in range(16):
            t = i * T / 16
            tidx = int(t/dt)
            k = int(i / 4)
            l = i % 4
            axs[k,l].plot(x, U_DNS[:,tidx], '-')

        print(f"Save {figName}")
        fig.savefig(figName)
        plt.close()


    if dump:
        print(f"Storing U_DNS {U_DNS.shape}")
        pickle.dump(U_DNS, open(f'{scratch}/DNS_Burgers_{ic}_s{s}_M{M}_N{N}.pickle'), 'wb')
        print(f"Storing f_store {f_store.shape}")
        pickle.dump(f_store, open(f'{scratch}/DNS_Force_{ic}_LES_s{s}_M{M}_N{N}.pickle'), 'wb')
        print(f"Storing u_bar {u_bar.shape}")
        np.save('{}/u_bar.npy'.format(scratch),u_bar)
        print(f"Storing f_bar {f_bar.shape}")
        np.save('{}/f_bar.npy'.format(scratch),f_bar)
        print(f"Storing PI {PI.shape}")
        np.save('{}/PI.npy'.format(scratch),PI)



full_input = np.load( f'{scratch}/u_bar.npy')
full_input = full_input.T

full_output = np.load( f'{scratch}/PI.npy')
full_output = full_output.T


full_input, full_output = helpers.shift_data(full_input, full_output)


train_region = int(0.8*M)
norm_input, mean_input, std_input = helpers.normalize_data(full_input[:train_region,:])
norm_output, mean_output, std_output = helpers.normalize_data(full_output[:train_region,:])

training_input = norm_input
training_output = norm_output

print(f'shape of input {training_input.shape}')
print(f'shape of output {training_output.shape}')

print(f'std_input: {std_input}')
print(f'std_output: {std_output}')

print(f'mean_input: {mean_input}')
print(f'mean_output: {mean_output}')

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
model.fit(training_input, training_output, epochs=100, batch_size=200, shuffle=True, validation_split=0.2)

model.save_weights(f'{scratch}/weights_trained_ANN')
exit()
