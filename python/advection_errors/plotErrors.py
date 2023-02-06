import json
import matplotlib.pyplot as plt 

def readJson(fname):
    with open(fname, 'r') as openfile:
        json_object = json.load(openfile)
        return json_object

dump8 = readJson('error_8.json')
dump16 = readJson('error_16.json')
#dump32 = readJson('error_32.json')
dump128 = readJson('error_128.json')

import sys
import argparse
sys.path.append('./../_model/')

import numpy as np
from Advection import *
from advection_environment_simple import setup_dns_default

colors = ['red', 'blue', 'magenta', 'green']

tt = dump8['t']
fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
ax.set_yscale('log')
ax.plot(tt, dump8['mse'], label='N8', color=colors[0])
ax.plot(tt, dump16['mse'], label='N16', color=colors[1])
#ax.plot(tt, dump32['mse'], label='N32', color=colors[2])
ax.plot(tt, dump128['mse'], label='N128', color=colors[3])


episodeLength = len(tt)
nu = 0.5
T = 2*np.pi/nu
dt = T / 200

famse = []
falinf = []
famass = []
for i, nx in enumerate([8, 16, 32, 128]):
    fa = Advection(N=nx, dt=dt, nu=nu, case='sinus', tend=T)
    fa.simulate()
    
    mse = np.mean((fa.uu - fa.solution)**2, axis=1)
    linf = np.amax(np.abs(fa.uu - fa.solution), axis=1)
    mass = np.sum(fa.uu, axis=1)
    
    famse.append(mse)
    falinf.append(linf)
    famass.append(mass)

for i, nx in enumerate([8, 16, 32, 128]):
    ax.plot(tt, famse[i], color=colors[i],linestyle='--')

ax.legend()

plt.tight_layout()
plt.savefig('mse_errors.pdf')
plt.close()


fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
ax.set_yscale('log')
ax.plot(tt, dump8['linf'], color=colors[0])
ax.plot(tt, dump16['linf'], color=colors[1])
#ax.plot(tt, dump32['linf'], color=colors[2])
ax.plot(tt, dump128['linf'], color=colors[3])

for i, nx in enumerate([8, 16, 32, 128]):
    ax.plot(tt, falinf[i], color=colors[i],linestyle='--')


plt.tight_layout()
plt.savefig('inf_errors.pdf')
plt.close()

fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(6,6))
ax.plot(tt, dump8['mass'], color=colors[0])
ax.plot(tt, dump16['mass'], color=colors[1])
#ax.plot(tt, dump32['mass'], color=colors[2])
ax.plot(tt, dump128['mass'], color=colors[3])

for i, nx in enumerate([8, 16, 32, 128]):
    ax.plot(tt, famass[i], color=colors[i],linestyle='--')


plt.tight_layout()
plt.savefig('mass_errors.pdf')
plt.close()
