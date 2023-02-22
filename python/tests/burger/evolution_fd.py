#!/bin/python3

# Discretization grid
N = 1024
N2 = 256

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse
sys.path.append('./../../_model/')

import numpy as np
from plotting import *
from Burger import *
from Burger_fd import *

#------------------------------------------------------------------------------
## set parameters and initialize simulation
L       = 2*pi
dt      = 0.001
s       = 1
tEnd    = 5
ic      = 'turbulence'
forcing = False

ssm = False
dsm = True

#L       = 100
#dt      = 0.01
#s       = 20
#tEnd    = 5000
#ic      = 'sinus'
#forcing = True

nu      = 0.02
#ic      = 'zero'
#ic      = 'turbulence'
#ic      = 'forced'
noise   = 0.
seed    = 42
#forcing = False


numruns = 100
errors = np.zeros((numruns,N2//2))
i = 0
while(i < numruns):
    print(f"run {i}/{numruns}")
    seed += 1
    dns = Burger(L=L, N=N, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed, forcing=forcing, s=s)
    dns.simulate()
    dns.compute_Ek()

    u_restart = dns.uu[0,:].copy()
    f_restart = interpolate.interp1d(dns.x, u_restart)

    sgs = Burger_fd(L=L, N=N2, dt=dt, nu=nu, tend=tEnd, case=ic, noise=noise, seed=seed+i, forcing=forcing, s=s, ssm=ssm, dsm=dsm)

    u0 = f_restart(sgs.x)
    sgs.IC( u0 = u0)

    sgs.setGroundTruth(dns.x, dns.tt, dns.uu)
    sgs.randfac1 = dns.randfac1
    sgs.randfac2 = dns.randfac2

    #------------------------------------------------------------------------------
    print("Simulate SGS..")
    ## simulate
    sgs.simulate()
    # convert to physical space
    sgs.compute_Ek()

    #------------------------------------------------------------------------------
    try:
        err = (np.abs(dns.Ek_ktt[dns.ioutnum,:N2//2] - sgs.Ek_ktt[dns.ioutnum,:N2//2])/dns.Ek_ktt[dns.ioutnum,:N2//2])**2
        kRelErr = np.mean(err)
        print(f"Relative spectal reward {kRelErr}")

        ## plot
        #plotField([dns, sgs])
        #plotAvgSpectrum([dns, sgs])
        #plotError(dns, sgs)
        #makePlot(dns, sgs, sgs, "evolution_fd", True)
        errors[i,:] = err
        i += 1
    except:
        print(f"excpetion thrown")


uq = np.quantile(errors, axis=0, q=0.8)
lq = np.quantile(errors, axis=0, q=0.2)
me = np.quantile(errors, axis=0, q=0.5)

plt.plot(me, color='coral')
plt.fill_between(np.arange(0,N2//2), uq, lq, color='coral', alpha=0.2)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig("quantiles_dsm.pdf")
plt.close()







