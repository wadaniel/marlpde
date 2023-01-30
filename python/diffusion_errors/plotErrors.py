import json
import matplotlib.pyplot as plt 

def readJson(fname):
    with open(fname, 'r') as openfile:
        json_object = json.load(openfile)
        return json_object

dump8 = readJson('error_8.json')
dump16 = readJson('error_16.json')
dump32 = readJson('error_32.json')
dump128 = readJson('error_128.json')

import sys
import argparse
sys.path.append('./../_model/')

import numpy as np
from Diffusion import *
from diffusion_environment import setup_dns_default

fdmse = []
fdlind = []
colors = ['red', 'blue', 'magenta', 'green']

tt = dump8['t']
fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
ax.set_yscale('log')
ax.plot(tt, dump8['mse'], label='N8', color=colors[0])
ax.plot(tt, dump16['mse'], label='N16', color=colors[1])
ax.plot(tt, dump32['mse'], label='N32', color=colors[2])
ax.plot(tt, dump128['mse'], label='N128', color=colors[3])

for i, nx in enumerate([8, 16, 32, 128]):
    fd = Diffusion(N=nx, dt=0.01, nu=0.1, case='sinus', tend=3)
    fd.simulate()
    
    mse = np.mean((fd.uu - fd.solution)**2, axis=1)
    ax.plot(tt, mse, color=colors[i],linestyle='--')


ax.legend()

plt.tight_layout()
plt.savefig('mse_errors.pdf')
plt.close()


tt = dump8['t']
fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
ax.set_yscale('log')
ax.plot(tt, dump8['linf'])
ax.plot(tt, dump16['linf'])
ax.plot(tt, dump32['linf'])
ax.plot(tt, dump128['linf'])

for i, nx in enumerate([8, 16, 32, 128]):
    fd = Diffusion(N=nx, dt=0.01, nu=0.1, case='sinus', tend=3)
    fd.simulate()
    
    linf = np.amax(np.abs(fd.uu - fd.solution), axis=1)
    ax.plot(tt, linf, color=colors[i],linestyle='--')


plt.tight_layout()
plt.savefig('inf_errors.pdf')
plt.close()


