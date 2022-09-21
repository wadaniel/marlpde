import matplotlib as mpl
mpl.use('Agg')

import os
import sys
from tqdm import trange
from Burger import Burger

#import h5py
import pickle
import helpers
from helpers import swish
import numpy as np
import matplotlib.pyplot as plt

scratch = os.getenv("SCRATCH", default=".")
basedir = f'{scratch}/ddp'
if not os.path.exists(basedir):
    os.mkdir(basedir)

M=int(1e6)

N=1024        # grid size / num Fourier modes
N_bar=32     # sgs grid size / num Fourier modes
nu=0.02       # viscosity 
noise=0       # noise for ic
seed=42       # random seed
fseed=42      # random seed forcing
forcing=True  # apply forcing term during step
s=20          # ratio of LES and DNS time steps

L  = 100        # domainsize
dt = 0.001      # time step
T  = 1000       # terminal time
ic = "sinus"    # initial condition

"""
N=1024          # grid size / num Fourier modes
N_bar=128       # sgs grid size / num Fourier modes
nu=0.02         # viscosity 
noise=0         # noise for ic
seed=42         # random seed ic
fseed=42        # random seed forcing
forcing=False   # apply forcing term during step
s=1             # ratio of LES and DNS time steps

L  = 2*np.pi    # domainsize
dt = 0.001      # time step
T  = 5          # terminal time
ic = "turbulence"   # initial condition
"""

run=0
figName = f"evolution_{ic}_{N}_{run}.pdf"

nunoise=False

dump=True# store fields
plot=True# create plot
simulate=True# create new dns

# domain discretization
x = np.arange(N)/L
# Storage for DNS field
U_DNS=np.zeros((N,M), dtype=np.float16)
# Storage for forcing terms
f_store=np.zeros((N,M), dtype=np.float16)

ns = int(T/dt)
nm = int(M/ns)
print(f'num steps {ns}, num simulations {nm}')

if (simulate == True):
    for i in trange(nm):
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
                s=s,
                nunoise=nunoise)

        dns.simulate()
        
        U_DNS[:,i*ns:(i+1)*ns] = np.transpose(dns.uu[:-1,:])
        f_store[:,i*ns:(i+1)*ns] = np.transpose(dns.f[:-1,:])
else:
    U_DNS = np.load( f'{basedir}/u_bar_{ic}_{N}_{N_bar}_{run}.npy')
    print(f"Loaded U_DNS: {U_DNS.shape}")
    f_store = np.load( f'{basedir}/f_bar_{ic}_{N}_{N_bar}_{run}.npy' )
    print(f"Loaded f_store: {f_store.shape}")

u_bar, PI, f_bar = helpers.calc_bar(U_DNS, f_store, N, N_bar, L)

if dump:
    print(f"Storing U_DNS {U_DNS.shape}")
    pickle.dump(U_DNS, open(f'{basedir}/DNS_Burgers_{ic}_s{s}_M{M}_N{N}_{run}.pickle', 'wb'))
    print(f"Storing f_store {f_store.shape}")
    pickle.dump(f_store, open(f'{basedir}/DNS_Force_{ic}_LES_s{s}_M{M}_N{N}_{run}.pickle', 'wb'))
    print(f"Storing u_bar {u_bar.shape}")
    np.save(f'{basedir}/u_bar_{ic}_{N}_{N_bar}_{run}.npy',u_bar)
    print(f"Storing f_bar {f_bar.shape}")
    np.save(f'{basedir}/f_bar_{ic}_{N}_{N_bar}_{run}.npy',f_bar)
    print(f"Storing PI {PI.shape}")
    np.save(f'{basedir}/PI_{ic}_{N}_{N_bar}_{run}.npy',PI)

if (plot == True):
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
