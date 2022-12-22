import time
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy import interpolate
import matplotlib.pyplot as plt

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



if __name__ == "__main__":

    L    = 2*np.pi
    tEnd = 5
    nu   = 0.02

    NDNS   = 2048
    dtNDNS = 0.0001

    xdns, tdns, dns_rk3, _ = simulate_burger_rk3(NDNS,L,dtNDNS,tEnd,nu)
    xdns, tdns, dns_abcn, _ = simulate_burger_abcn(NDNS,L,dtNDNS,tEnd,nu)

    fdns_rk3 = interpolate.interp2d(xdns, tdns, dns_rk3, kind='cubic')
    fdns_abcn = interpolate.interp2d(xdns, tdns, dns_abcn, kind='cubic')

    ############## find spatial error ##############################
    I = 5
    N_i = 512
    dt = dtNDNS
    mserk3 = np.zeros(I)
    mseaabcn =  np.zeros(I)
    Narr = np.zeros(I)

    for i in range(0,I):
        x, t, sol_rk3, trk3 = simulate_burger_rk3(N_i,L,dt,tEnd,nu)
        print("break1")
        print(sol_rk3)
        x, t, sol_abcn, tabcn = simulate_burger_abcn(N_i,L,dt,tEnd,nu)
        print("break2")
        print(sol_abcn)
        mserk3[i] = np.mean(np.absolute(sol_rk3 - fdns_rk3(x, t)))
        mseaabcn[i] = np.mean(np.absolute(sol_abcn - fdns_abcn(x, t)))
        Narr[i] = N_i
        N_i = int(N_i/2)

    harr = L/Narr
    print(mserk3)
    print(mseaabcn)

    figName = "err"

    p = 3

    fig, axs = plt.subplots(1,2, figsize=(15,15))
    axs[0].plot(harr, mserk3, '-', label='RK3')
    axs[0].plot(harr, mseaabcn, '--', label='ABCN')
    axs[0].plot(harr, np.power(harr, p))
    axs[0].set_xscale('log')
    axs[0].set_xlabel('dx')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Err')
    axs[0].legend()

    ############## find temporal error ##############################

    I = 3
    N = 1024
    dt_i = dtNDNS
    mserk3 = np.zeros(I)
    mseaabcn =  np.zeros(I)
    deltat = np.zeros(I)

    for i in range(0,I):
        x, t, sol_rk3, trk3 = simulate_burger_rk3(N,L,dt_i,tEnd,nu)
        x, t, sol_abcn, tabcn = simulate_burger_abcn(N,L,dt_i,tEnd,nu)
        mserk3[i] = np.mean(np.absolute(sol_rk3 - fdns_rk3(x, t)))
        mseaabcn[i] = np.mean(np.absolute(sol_abcn - fdns_abcn(x, t)))
        deltat[i] = dt_i
        dt_i = 2*dt_i

    print(mserk3)
    print(mseaabcn)

    p = 2

    axs[1].plot(deltat, mserk3, '-', label='RK3')
    axs[1].plot(deltat, mseaabcn, '--', label='ABCN')
    axs[1].plot(deltat, np.power(deltat, p))
    axs[1].set_xscale('log')
    axs[1].set_xlabel('dt')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Err')

    print(f"Save {figName}")
    fig.savefig(figName)
