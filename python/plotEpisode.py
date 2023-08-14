#!/bin/python3

# Discretization grid
N2 = 256

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.stats import gaussian_kde

fname = 'episodes_101'
import numpy as np

print(f"Loading file {fname}")
npzfile = np.load(f'{fname}.npz')
#dns_Ektt = np.vstack((npzfile['dns_Ektt'], dns.Ek_ktt))
#sgs_Ektt = np.vstack((npzfile['sgs_Ektt'], sgs.Ek_ktt))
errors = npzfile['err_t']
print(f'loaded errors {errors.shape}')


print("plot quantiles")
uq = np.quantile(errors, axis=0, q=0.8)
lq = np.quantile(errors, axis=0, q=0.2)
me = np.quantile(errors, axis=0, q=0.5)

plt.plot(me, color='coral')
plt.fill_between(np.arange(0,N2//2), uq, lq, color='coral', alpha=0.2)
plt.ylim(1e-4, 1e2)

plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig(f"quantiles_{fname}.pdf")
plt.close()


print("plot sgs")
sgs = npzfile['sgs_actions']
smin = -4. #sgs.min()
smax = 4. #sgs.max()
svals = np.linspace(smin,smax,500) 

sgsDensity = gaussian_kde(sgs.flatten())
sgsDensityVals = sgsDensity(svals)
plt.plot(svals, sgsDensityVals)
plt.yscale('log')
plt.tight_layout()
plt.savefig(f"psgs_{fname}.pdf")
plt.close()

#sgs_u = np.vstack((npzfile['sgs_u'], sgs.uu))
#dns_u = np.vstack((npzfile['dns_u'], dns.uu))


