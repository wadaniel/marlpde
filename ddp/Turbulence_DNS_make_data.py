import os
import sys
from tqdm import trange
sys.path.append('../python/_model')
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
                s=s,
                version=0, 
                nunoise=nunoise, 
                numAgents=1)

        dns.simulate()
        
        U_DNS[:,i*ns:(i+1)*ns] = np.transpose(dns.uu[:ns,:])
        f_store[:,i*ns:(i+1)*ns] = np.transpose(dns.f[:ns,:])
else:
    U_DNS = np.load( f'{basedir}/u_bar.npy')
    print(f"Loaded U_DNS: {U_DNS.shape}")
    f_store = np.load( f'{basedir}/f_bar.npy' )
    print(f"Loaded f_store: {f_store.shape}")

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
    pickle.dump(U_DNS, open(f'{basedir}/DNS_Burgers_{ic}_s{s}_M{M}_N{N}.pickle', 'wb'))
    print(f"Storing f_store {f_store.shape}")
    pickle.dump(f_store, open(f'{basedir}/DNS_Force_{ic}_LES_s{s}_M{M}_N{N}.pickle', 'wb'))
    print(f"Storing u_bar {u_bar.shape}")
    np.save('{}/u_bar.npy'.format(basedir),u_bar)
    print(f"Storing f_bar {f_bar.shape}")
    np.save('{}/f_bar.npy'.format(basedir),f_bar)
    print(f"Storing PI {PI.shape}")
    np.save('{}/PI.npy'.format(basedir),PI)
