import matplotlib as mpl
mpl.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde

scratch = os.getenv("SCRATCH", default=".")
basedir = f'{scratch}/ddp'
figNameU = f'ddp_plot_u.png'
figNamePiPrior = f'ddp_plot_pi_prior.png'
figNamePiPost = f'ddp_plot_pi_post.png'

u_bar = np.load( f'{basedir}/u_bar.npy')
print(f'shape of u_bar {u_bar.shape}')

ndata = int(1e5)
u_bar=u_bar[:,:ndata]

u_bar_flat = u_bar.flatten()
u_vals = np.linspace(min(u_bar_flat), max(u_bar_flat), 500)
u_bar_density = gaussian_kde(u_bar_flat)
u_bar_density_vals = u_bar_density(u_vals)

fig, axs = plt.subplots(1, 2) #, subplot_kw=dict(box_aspect=1), figsize=(10,10))
axs[0].plot(u_vals, u_bar_density_vals)

fac = 3
u_bar_mean = np.mean(u_bar_flat)
u_bar_sdev = np.std(u_bar_flat)
u_vals2 = np.linspace(u_bar_mean-fac*u_bar_sdev,u_bar_mean+fac*u_bar_sdev,500)
axs[1].plot(u_vals2, u_bar_density(u_vals))
    
print(f"Save {figNameU}")
fig.savefig(figNameU)

pi_predict = np.load( f'{basedir}/PI_predict_prior.npy')
print(f'shape of pi_predict {pi_predict.shape}')
pi_predict = pi_predict[:,:ndata]
pi_predict_flat = pi_predict.flatten()

pi_vals = np.linspace(min(pi_predict_flat), max(pi_predict_flat), 500)
pi_density = gaussian_kde(pi_predict_flat)
pi_density_vals = pi_density(pi_vals)

fig, axs = plt.subplots(1, 2)
axs[0].plot(pi_vals, pi_density_vals)

fac = 3
pi_mean = np.mean(pi_predict_flat)
pi_sdev = np.std(pi_predict_flat)
pi_vals2 = np.linspace(pi_mean-fac*pi_sdev, pi_mean+fac*pi_sdev, 500)

axs[1].plot(pi_vals2, pi_density(pi_vals2))
    
print(f"Save {figNamePiPrior}")
fig.savefig(figNamePiPrior)

pi_predict = np.load( f'{basedir}/PI_predict_posterior.npy')
print(f'shape of pi_predict {pi_predict.shape}')
pi_predict = pi_predict[:,:ndata]
pi_predict_flat = pi_predict.flatten()

pi_vals = np.linspace(min(pi_predict_flat), max(pi_predict_flat), 500)
pi_density = gaussian_kde(pi_predict_flat)
pi_density_vals = pi_density(pi_vals)

fig, axs = plt.subplots(1, 2)
axs[0].plot(pi_vals, pi_density_vals)

fac = 3
pi_mean = np.mean(pi_predict_flat)
pi_sdev = np.std(pi_predict_flat)
pi_vals2 = np.linspace(pi_mean-fac*pi_sdev, pi_mean+fac*pi_sdev, 500)

axs[1].plot(pi_vals2, pi_density(pi_vals2))
    
print(f"Save {figNamePiPost}")
fig.savefig(figNamePiPost)
