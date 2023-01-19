import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate
from scipy.stats import gaussian_kde

def makePlot(dns, base, sgs, fileName, spectralReward=True):

#------------------------------------------------------------------------------

    figName2 = fileName + "_evolution.pdf"
    print(f"[plotting] Plotting {figName2} ...")

    tEnd = dns.tend
    dt = dns.dt
    sgs_dt = sgs.dt

    # step factor
    s = dns.stepper
    assert base.stepper == s
    assert sgs.stepper == s

    colors = ['black','royalblue','seagreen']
    fig2, axs2 = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
    for i in range(16):
        t = i * tEnd / 16
        tidx = int(t/dt)
        tidx_sgs = int(t/sgs_dt)
        k = int(i / 4)
        l = i % 4

        axs2[k,l].plot(base.x, base.uu[tidx_sgs,:], '-', color=colors[1])
        axs2[k,l].plot(sgs.x, sgs.uu[tidx_sgs,:], '-', color=colors[2])
        axs2[k,l].plot(dns.x, dns.uu[tidx,:], '--', color=colors[0])

    fig2.tight_layout()
    fig2.savefig(figName2)

#------------------------------------------------------------------------------

    N  = dns.N
    gridSize = sgs.N
    numActions = sgs.M

    nt = int(tEnd/dt)
    k1 = dns.k[:N//2]
    k2 = sgs.k[1:gridSize//2]

    #colors = plt.cm.jet(np.linspace(0,1,7))

    time = np.arange(tEnd/dt+1)*dt

    #fig1, axs1 = plt.subplots(3, 6, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(15,15))
    #fig1, axs1 = plt.subplots(3, 6, sharex='col', sharey='col', subplot_kw=dict(aspect=1.), figsize=(15,15))
    fig1, axs1 = plt.subplots(3, 6, sharex='col', sharey='col', figsize=(15,15))

    umax = max(dns.uu.max(), base.uu.max(), sgs.uu.max())
    umin = min(dns.uu.min(), base.uu.min(), sgs.uu.min())
    ulevels = np.linspace(umin, umax, 50)

#------------------------------------------------------------------------------
    print("[plotting] plot DNS")

    idx = 0
    axs1[0,0].contourf(dns.x, dns.tt, dns.uu, ulevels)

    axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[0,0:N//2]),':', color=colors[idx])
    axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[nt//2,0:N//2]),'--', color=colors[idx])
    axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[-1,0:N//2]),'-', color=colors[idx])
    axs1[0,3].plot(k1[2:-10], 1e-5*k1[2:-10]**(-2),'--', linewidth=0.5)
    axs1[0,3].set_xscale('log')
    axs1[0,3].set_yscale('log')

#------------------------------------------------------------------------------
    try:

        print(dns.Ek_ktt.shape)
        tidx = np.arange(start=0,stop=nt+1,step=dns.stepper)

        f_dns = interpolate.interp2d(dns.x, dns.tt, dns.uu, kind='cubic')
        udns_int = f_dns(base.x, base.tt)
        errBaseU = np.abs(base.uu-udns_int)
        mseBaseU_t = np.mean(errBaseU**2, axis=1)
        mseBaseU = np.cumsum(mseBaseU_t)/np.arange(1, len(mseBaseU_t)+1)

        errBaseK_t = np.mean(((np.abs(dns.Ek_ktt[tidx,1:gridSize//2] - base.Ek_ktt[:,1:gridSize//2])/dns.Ek_ktt[tidx,1:gridSize//2]))**2, axis=1)
        errBaseK = np.cumsum(errBaseK_t)/np.arange(1, len(errBaseK_t)+1)

        udns_int = f_dns(sgs.x, sgs.tt)
        errU = np.abs(sgs.uu-udns_int)
        mseU_t = np.mean(errU**2, axis=1)
        mseU = np.cumsum(mseU_t)/np.arange(1, len(mseU_t)+1)

        errK_t = np.mean(((np.abs(dns.Ek_ktt[tidx,1:gridSize//2] - sgs.Ek_ktt[:,1:gridSize//2])/dns.Ek_ktt[tidx,1:gridSize//2]))**2, axis=1)
        errK = np.cumsum(errK_t)/np.arange(1, len(errK_t)+1)

#------------------------------------------------------------------------------

        emax = max(errBaseU.max(), errU.max())
        emin = min(errBaseU.min(), errU.min())
        elevels = np.linspace(emin, emax, 50)

#------------------------------------------------------------------------------

        print("[plotting] plot baseline")
        idx = idx + 1

        # Plot solution
        axs1[idx,0].contourf(base.x, base.tt, base.uu, ulevels)

        # Plot difference to dns
        axs1[idx,1].contourf(base.x, base.tt, errBaseU, elevels)

        # Plot instanteneous spec err and cumulative spec err
        if spectralReward:
            axs1[idx,2].plot(base.tt, errBaseK_t, 'r:')
            axs1[idx,2].plot(base.tt, errBaseK, 'r-')

        # Plot instanteneous mse and cumulative mse
        else:
            axs1[idx,2].plot(base.tt, mseBaseU_t, 'r:')
            axs1[idx,2].plot(base.tt, mseBaseU, 'r-')

        axs1[idx,2].set_yscale('log')
        axs1[idx,2].set_ylim([1e-4,1e1])

        # Plot energy spectrum at start, mid and end of simulation
        axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[0,1:gridSize//2]),':',color=colors[idx])
        axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[nt//(2*s),1:gridSize//2]),'--',color=colors[idx])
        axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[-1,1:gridSize//2]),'-',color=colors[idx])
        axs1[idx,3].set_xscale('log')
        axs1[idx,3].set_yscale('log')
        axs1[idx,3].set_ylim([1e-4,None])

        # Plot energy spectrum difference
        axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[0,1:gridSize//2] - base.Ek_ktt[0,1:gridSize//2])/dns.Ek_ktt[0,1:gridSize//2]),'r:')
        axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[nt//2,1:gridSize//2] - base.Ek_ktt[nt//(2*s),1:gridSize//2])/dns.Ek_ktt[nt//2,1:gridSize//2]),'r--')
        axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[-1,1:gridSize//2] - base.Ek_ktt[-1,1:gridSize//2])/dns.Ek_ktt[-1,1:gridSize//2]),'r-')
        print(np.mean(np.abs((dns.Ek_ktt[-1,1:gridSize//2] - base.Ek_ktt[-1,1:gridSize//2])/dns.Ek_ktt[-1,1:gridSize//2])**2))

        axs1[idx,4].set_xscale('log')
        axs1[idx,4].set_yscale('log')
        #axs1[idx,4].set_ylim([1e-4,None])

#------------------------------------------------------------------------------

        print("[plotting] plot sgs")
        idx = idx + 1

        # Plot solution
        axs1[idx,0].contourf(sgs.x, sgs.tt, sgs.uu, ulevels)

        # Plot difference to dns
        axs1[idx,1].contourf(sgs.x, sgs.tt, errU, elevels)

        # Plot instanteneous spec err and cumulative spec err
        if spectralReward:
            axs1[idx,2].plot(base.tt, errK_t, 'r:')
            axs1[idx,2].plot(base.tt, errK, 'r-')

        else:
            # Plot instanteneous energy and time averaged energy
            axs1[idx,2].plot(sgs.tt, mseU, 'r:')
            axs1[idx,2].plot(sgs.tt, mseU_t, 'r-')

        # Plot time averaged energy spectrum at start, mid and end of simulation
        axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[0,1:gridSize//2]),':',color=colors[idx])
        axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[nt//(2*s),1:gridSize//2]),'--',color=colors[idx])
        axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[-1,1:gridSize//2]),'-',color=colors[idx])

        # Plot time averaged energy spectrum difference
        axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[0,1:gridSize//2] - sgs.Ek_ktt[0,1:gridSize//2])/dns.Ek_ktt[0,1:gridSize//2]),'r:')
        axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[nt//2,1:gridSize//2] - sgs.Ek_ktt[nt//(2*s),1:gridSize//2])/dns.Ek_ktt[nt//2,1:gridSize//2]),'r--')
        axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[-1,1:gridSize//2] - sgs.Ek_ktt[-1,1:gridSize//2])/dns.Ek_ktt[-1,1:gridSize//2]),'r-')
        print(np.mean(np.abs((dns.Ek_ktt[-1,1:gridSize//2] - sgs.Ek_ktt[-1,1:gridSize//2])/dns.Ek_ktt[-1,1:gridSize//2])**2))

        actioncolors = plt.cm.coolwarm(np.linspace(0,1,numActions))
        for i in range(numActions):
            axs1[idx,5].plot(sgs.tt[1:], sgs.actionHistory[1:,i], color=actioncolors[i])

        plt.tight_layout()
        figName = fileName + ".png"
        print(f"[plotting] Plotting {figName} ...")
        fig1.savefig(figName)

    except Exception as e:
        print("[plotting] Exception during plotting:")
        print(e)

#------------------------------------------------------------------------------
    plt.close('all')

    print("[plotting] plot actions..")
    try:
        figName3 = fileName + "_action.png"

        sgsHistory = sgs.sgsHistory.flatten()
        sgsHistory = sgsHistory[np.abs(sgsHistory)<10]

        smax = sgsHistory.max()
        smin = sgsHistory.min()
        slevels = np.linspace(smin, smax, 50)
        svals = np.linspace(smin, smax, 500)

        fig3, axs3 = plt.subplots(1, 3, sharex='col', sharey='col', figsize=(10,10))

        axs3[0].contourf(sgs.x, sgs.tt, sgs.sgsHistory)

        urs_sDensity = gaussian_kde(sgsHistory
        urs_sDensityVals = urs_sDensity(svals)
        axs3[1].plot(svals, urs_sDensityVals)
        axs3[1].set_ylim([1e-5, 100])
        axs3[1].set_yscale('log')

        sfac = 3
        urs_sMean = np.mean(sgs.sgsHistory)
        print("static mean")
        print(urs_sMean)
        urs_sSdev = np.std(sgs.sgsHistory)
        print("static deviation")
        print(urs_sSdev)
        svals2  = np.linspace(urs_sMean-sfac*urs_sSdev,urs_sMean+sfac*urs_sSdev,500)
        axs3[2].plot(svals2, urs_sDensity(svals2))
        axs3[2].set_ylim([1e-5, 100])
        axs3[2].set_yscale('log')

        print("Plotting {} ...".format(figName3))
        fig3.savefig(figName3)

    except Exception as e:
        print("[plotting] Exception during plotting:")
        print(e)

    plt.close('all')
