import matplotlib.pyplot as plt 
import numpy as np
from scipy import interpolate

def makePlot(dns, base, sgs, fileName):
      
    N  = dns.N
    gridSize = sgs.N
    dt = dns.dt
    tEnd = dns.tend
    numActions = sgs.M

    nt = int(tEnd/dt)
    k1 = dns.k[:N//2]
    k2 = sgs.k[:gridSize//2]

    colors = plt.cm.jet(np.linspace(0,1,5))

    time = np.arange(tEnd/dt+1)*dt

    fig1, axs1 = plt.subplots(3, 6, sharex='col', sharey='col', subplot_kw=dict(box_aspect=1), figsize=(15,15))

    umax = max(dns.uu.max(), base.uu.max(), sgs.uu.max())
    umin = min(dns.uu.min(), base.uu.min(), sgs.uu.min())
    ulevels = np.linspace(umin, umax, 50)

#------------------------------------------------------------------------------
    print("plot DNS")

    idx = 0
    axs1[0,0].contourf(dns.x, dns.tt, dns.uu, ulevels)

    axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[0,0:N//2]),':', color=colors[idx])
    axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[nt//2,0:N//2]),'--', color=colors[idx])
    axs1[0,3].plot(k1, np.abs(dns.Ek_ktt[-1,0:N//2]),'-', color=colors[idx])
    axs1[0,3].plot(k1[2:-10], 1e-5*k1[2:-10]**(-2),'--', linewidth=0.5)
    axs1[0,3].set_xscale('log')
    axs1[0,3].set_yscale('log')

#------------------------------------------------------------------------------

    errBaseEk_t = dns.Ek_t - base.Ek_t
    errBaseEk_tt = dns.Ek_tt - base.Ek_tt

    f_dns = interpolate.interp2d(dns.x, dns.tt, dns.uu, kind='cubic')
    udns_int = f_dns(base.x, base.tt)
    errBaseU = np.abs(base.uu-udns_int)
    mseBaseU_t = np.mean(errBaseU**2, axis=1)
    
    errEk_t = dns.Ek_t - sgs.Ek_t
    errEk_tt = dns.Ek_tt - sgs.Ek_tt
    
    errU = np.abs(sgs.uu-udns_int)
    mseU_t = np.mean(errU**2, axis=1)

#------------------------------------------------------------------------------

    emax = max(errBaseU.max(), errU.max())
    emin = min(errBaseU.min(), errU.min())
    elevels = np.linspace(emin, emax, 50)
    
#------------------------------------------------------------------------------
    print("plot baseline")
    idx = idx + 1

    # Plot solution
    axs1[idx,0].contourf(base.x, base.tt, base.uu, ulevels)

    # Plot difference to dns
    axs1[idx,1].contourf(base.x, base.tt, errBaseU, elevels)

    # Plot instanteneous energy and time averaged energy
    axs1[idx,2].plot(base.tt, mseBaseU_t, 'r-')
    axs1[idx,2].set_yscale('log')
    #axs1[idx,2].set_ylim([1e-8,None])

    # Plot energy spectrum at start, mid and end of simulation
    axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[0,0:gridSize//2]),':',color=colors[idx])
    axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[nt//2,0:gridSize//2]),'--',color=colors[idx])
    axs1[idx,3].plot(k2, np.abs(base.Ek_ktt[-1,0:gridSize//2]),'-',color=colors[idx])
    axs1[idx,3].set_xscale('log')
    axs1[idx,3].set_yscale('log')
    axs1[idx,3].set_ylim([1e-3,None])

    # Plot energy spectrum difference
    axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[0,:gridSize//2] - base.Ek_ktt[0,:gridSize//2])/dns.Ek_ktt[0,:gridSize//2]),'r:')
    #axs1[idx,4].plot(k2[1:], np.abs(dns.Ek_ktt[0,1:gridSize//2] - base.Ek_ktt[0,1:gridSize//2]),'r:')
    axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[nt//2,:gridSize//2] - base.Ek_ktt[nt//2,:gridSize//2])/dns.Ek_ktt[nt//2,:gridSize//2]),'r--')
    #axs1[idx,4].plot(k2, np.abs(dns.Ek_ktt[nt//2,0:gridSize//2] - base.Ek_ktt[gridSize//2,0:gridSize//2]),'r--')
    axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[-1,:gridSize//2] - base.Ek_ktt[-1,:gridSize//2])/dns.Ek_ktt[-1,:gridSize//2]),'r-')
    #axs1[idx,4].plot(k2, np.abs(dns.Ek_ktt[-1,0:gridSize//2] - base.Ek_ktt[-1,0:gridSize//2]),'r')
    axs1[idx,4].set_xscale('log')
    axs1[idx,4].set_yscale('log')
    #axs1[idx,4].set_ylim([1e-4,None])

#------------------------------------------------------------------------------

    print("plot sgs")
    idx = idx + 1

    # Plot solution
    axs1[idx,0].contourf(sgs.x, sgs.tt, sgs.uu, ulevels)

    # Plot difference to dns
    axs1[idx,1].contourf(sgs.x, sgs.tt, errU, elevels)

    # Plot instanteneous energy and time averaged energy
    axs1[idx,2].plot(sgs.tt, mseU_t, 'r-')
    axs1[idx,2].set_yscale('log')

    # Plot time averaged energy spectrum at start, mid and end of simulation
    axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[0,0:gridSize//2]),':',color=colors[idx])
    axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[nt//2,0:gridSize//2]),'--',color=colors[idx])
    axs1[idx,3].plot(k2, np.abs(sgs.Ek_ktt[-1,0:gridSize//2]),'-',color=colors[idx])
    axs1[idx,3].set_xscale('log')
    axs1[idx,3].set_yscale('log')

    # Plot time averaged energy spectrum difference
    axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[0,:gridSize//2] - sgs.Ek_ktt[0,:gridSize//2])/dns.Ek_ktt[0,:gridSize//2]),'r:')
    #axs1[idx,4].plot(k2[1:], np.abs(dns.Ek_ktt[0,1:gridSize//2] - sgs.Ek_ktt[0,1:gridSize//2]),'r:')
    axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[nt//2,:gridSize//2] - sgs.Ek_ktt[nt//2,:gridSize//2])/dns.Ek_ktt[nt//2,:gridSize//2]),'r--')
    #axs1[idx,4].plot(k2, np.abs(dns.Ek_ktt[nt//2,0:gridSize//2] - sgs.Ek_ktt[gridSize//2,0:gridSize//2]),'r--')
    axs1[idx,4].plot(k2, np.abs((dns.Ek_ktt[-1,:gridSize//2] - sgs.Ek_ktt[-1,:gridSize//2])/dns.Ek_ktt[-1,:gridSize//2]),'r-')
    #axs1[idx,4].plot(k2, np.abs(dns.Ek_ktt[-1,0:gridSize//2] - sgs.Ek_ktt[-1,0:gridSize//2]),'r')
    axs1[idx,4].set_xscale('log')
    axs1[idx,4].set_yscale('log')

    actioncolors = plt.cm.coolwarm(np.linspace(0,1,numActions))
    for i in range(numActions):
        axs1[idx,5].plot(sgs.tt[1:], sgs.actionHistory[:,i], color=actioncolors[i])

    plt.tight_layout()
    figName = fileName + ".png"
    fig1.savefig(figName)

#------------------------------------------------------------------------------

    figName2 = fileName + "_evolution.pdf"
    print("Plotting {} ...".format(figName2))
    
    fig2, axs2 = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))
    for i in range(16):
        t = i * tEnd / 16
        tidx = int(t/dt)
        k = int(i / 4)
        l = i % 4
        
        axs2[k,l].plot(base.x, base.uu[tidx,:], '--', color=colors[1])
        axs2[k,l].plot(sgs.x, sgs.uu[tidx,:], '-', color=colors[2])
        axs2[k,l].plot(dns.x, dns.uu[tidx,:], '--', color=colors[0])

    fig2.savefig(figName2)
