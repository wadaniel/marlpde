import time
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy import interpolate

def simulate_burger_rk3(N, L, dt, tEnd, nu):
    
    tic = time.perf_counter()
     
    x   = np.arange(0,N) * L/N
    icx = np.sin(x)
    
    nsteps = int(tEnd/dt)
    t   = np.arange(0, nsteps+1) * dt

    k   = fftfreq(N, L / (2*np.pi*N))
    k1  = 1j * k
    k2  = k1**2
 
    u = icx
    v = fft(u)

    uu = np.zeros((nsteps+1,N))
    uu[0,:] = u

    tc = 0
    for i in range(1, nsteps+1):
        v1 = v + dt * (-0.5*k1*fft(u**2) + nu*k2*v)
        u1 = np.real(ifft(v1))
        
        v2 = 3./4.*v + 1./4.*v1 + 1./4. * dt * (-0.5*k1*fft(u1**2) + nu*k2*v1)
        u2 = np.real(ifft(v2))

        v3 = 1./3.*v + 2./3.*v2 + 2./3. * dt * (-0.5*k1*fft(u2**2) + nu*k2*v2)
        v = v3
        
        u = np.real(ifft(v))
        uu[i,:] = u

        tc += dt

    #assert tc == tEnd, f"{tc} {tEnd}"
    toc = time.perf_counter()
    return x, t, uu, toc-tic


def simulate_burger_abcn(N, L, dt, tEnd, nu):

    tic = time.perf_counter()

    x   = np.arange(0,N) * L/N
    icx = np.sin(x)
    
    nsteps = int(tEnd/dt)
    t   = np.arange(0,nsteps+1) * dt

    k   = fftfreq(N, L / (2*np.pi*N))
    k1  = 1j * k
    k2  = k1**2
   
    u = icx
    v = fft(u)

    nsteps = int(tEnd/dt)
    uu = np.zeros((nsteps+1,N))
    uu[0,:] = u
    Fn_old=k1*fft(0.5*u**2)

    tc = 0
    for i in range(1, nsteps+1):
        
        C=-0.5*k2*nu*dt
        Fn=k1*fft(0.5*u**2)
        v=((1.0-C)*v-0.5*dt*(3.0*Fn-Fn_old))/(1.0+C)
        Fn_old = Fn.copy()
 
        u = np.real(ifft(v))
        uu[i,:] = u

        tc += dt

    #assert tc == tEnd, f"{tc} {tEnd}"
    toc = time.perf_counter()
    return x, t, uu, toc-tic


def plot_evolution(x, reslist, tEnd):
    import matplotlib.pyplot as plt
    
    figName = "evolution.pdf"
    print("Plotting {} ...".format(figName))
    
    fig, axs = plt.subplots(4,4, sharex=True, sharey=True, figsize=(15,15))

    for res in reslist:
        for i in range(16):
            t = i * tEnd / 16
            tidx = int(t/dt)
            k = int(i / 4)
            l = i % 4
            
            axs[k,l].plot(x, res[tidx,:], '-') #, color=colors[1])

    print(f"Save {figName}")
    fig.savefig(figName)


if __name__ == "__main__":
    
    L    = 2*np.pi
    tEnd = 5
    nu   = 0.02

    NDNS   = 1024
    dtNDNS = 0.0001
   
    xdns, tdns, dns_rk3, _ = simulate_burger_rk3(NDNS,L,dtNDNS,tEnd,nu)
    xdns, tdns, dns_abcn, _ = simulate_burger_abcn(NDNS,L,dtNDNS,tEnd,nu)

    fdns_rk3 = interpolate.interp2d(xdns, tdns, dns_rk3, kind='cubic')
    fdns_abcn = interpolate.interp2d(xdns, tdns, dns_abcn, kind='cubic')

    N    = 32
    dt   = 0.001

    x, t, sol_rk3, trk3 = simulate_burger_rk3(N,L,dt,tEnd,nu)
    mse = np.mean((sol_rk3 - 0.5*fdns_rk3(x, t) - 0.5*fdns_abcn(x, t))**2)
    print(sol_rk3)
    print(mse)
    print(trk3)
    
    x, t, sol_abcn, tabcn = simulate_burger_abcn(N,L,dt,tEnd,nu)
    mse = np.mean((sol_abcn - 0.5*fdns_rk3(x, t) - 0.5*fdns_abcn(x, t))**2)
    print(t)
    print(sol_abcn)
    print(mse)
    print(tabcn)

    #plot_evolution(x, [sol_rk3, sol_abcn], tEnd)
