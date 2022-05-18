import sys
from numpy import pi
from scipy import interpolate
from scipy.fftpack import fft, ifft, fftfreq
import numpy as np


np.seterr(over='raise', invalid='raise')

def gaussian( x, mean, sigma ):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-1/2*( (x-mean)/sigma )**2)

def hat( x, mean, dx ):
    left  = np.clip((x + dx - mean)/dx, a_min = 0., a_max = 1.)
    right = np.clip((dx - x + mean)/dx, a_min = 0., a_max = 1.)
    return left + right - 1.

class Kuramoto:
    #
    # Solution of the  KS equation
    #
    # u_t + nu*u_xx + u_xxxx + nu*0.5(u*u)_x = 0 (KS)
    # u_t + u_xx + 0.5(u*u)_x = 0 (VB)
    # with periodic BCs on x \in [0, 2*pi*L]: u(x+2*pi*L,t) = u(x,t).
    #
    # The nature of the solution depends on the system size L and on the initial
    # condition u(x,0).
    #
    # see P Cvitanović, RL Davidchack, and E Siminos, SIAM Journal on Applied Dynamical Systems 2010
    #
    # Spatial  discretization: spectral (Fourier)
    # Temporal discretization: exponential time differencing fourth-order Runge-Kutta
    # see AK Kassam and LN Trefethen, SISC 2005

    def __init__(self, L=2.*np.pi, N=512, dt=0.001, nu=1.0, nsteps=None, tend=5., u0=None, v0=None, case=None, noise=0., seed=42, ssm=False, dsm=False):

        # Initialize
        np.random.seed(None)
        self.noise = noise
        self.seed = seed

        self.L  = float(L);
        self.dt = float(dt);
        self.tend = float(tend)

        if (nsteps is None):
            nsteps = int(tend/dt)
        else:
            nsteps = int(nsteps)
            # override tend
            tend = dt*nsteps

        # save to self
        self.N      = N
        self.dx     = L/N
        self.x      = np.linspace(0, self.L, N, endpoint=False)
        self.dt     = dt
        self.nu     = nu
        self.nsteps = nsteps
        self.nout   = nsteps
        self.sigma  = L/(2*N)

        self.ssm = ssm
        self.dsm = dsm

        # determine sharp spectral filter
        if (dsm is True):
            self.Gker = np.zeros(N)
            self.dx_ = 2*self.dx # test scale width
            self.Gker[0] = 1
            for i in range(N-1):
                val = (i+1)*self.dx
                self.Gker[i+1] = np.sin(np.pi*val/self.dx_)/(np.pi*val)

        # time when field space transformed
        self.uut = -1
        # field in real space
        self.uu = None
        # ground truth in real space
        self.uu_truth = None
        # interpolation of truth
        self.f_truth = None

        # initialize simulation arrays
        self.__setup_timeseries()

        # set initial condition
        if (case is not None):
            self.IC(case=case)
        elif (u0 is None) and (v0 is None):
            self.IC()
        elif (u0 is not None):
            self.IC(u0 = u0)
        elif (v0 is not None):
            self.IC(v0 = v0)
        else:
            print("[KS] IC ambigous")
            sys.exit()

        # precompute Fourier-related quantities
        self.__setup_fourier()

        # precompute ETDRK4 scalar quantities:
        self.__setup_etdrk4()

    def __setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)

        # nout+1 because we store the IC as well
        self.uu = np.zeros([self.nout+1, self.N], dtype=np.complex64)
        self.vv = np.zeros([self.nout+1, self.N], dtype=np.complex64)
        self.tt = np.zeros(self.nout+1)
        self.sgsHistory = np.zeros([self.nout+1, self.N])
        self.tt[0]   = 0.

    def __setup_fourier(self, coeffs=None):
        # self.k = fftfreq(self.N, self.L / (2*np.pi*self.N))
        k1 = np.arange(0, self.N/2 - 0.5, 1)
        k2 = np.arange(-self.N/2+1, -0.5, 1)
        k3 = np.zeros(1)
        self.k = np.concatenate([k1, k3, k2])*(2*np.pi/self.L)
        #print(self.k)
        #print(self.k.shape)
        print(np.amax(self.k))
        # Fourier multipliers for the linear term Lu
        if (coeffs is None):
            # normal-form equation
            self.l = self.nu*self.k**2 - self.k**4 #(KS)
            #self.l = self.k**2 #(VB)
        else:
            # altered-coefficients
            self.l = -      coeffs[0]*np.ones(self.k.shape) \
                     -      coeffs[1]*1j*self.k             \
                     + (1 + coeffs[2])  *self.k**2          \
                     +      coeffs[3]*1j*self.k**3          \
                     - (1 + coeffs[4])  *self.k**4


    def __setup_etdrk4(self):
        self.E  = np.exp(self.dt*self.l)
        self.E2 = np.exp(self.dt*self.l/2.)
        self.M  = 62                                           # no. of points for complex means
        self.r  = np.exp(1j*pi*(np.r_[1:self.M+1]-0.5)/self.M) # roots of unity
        self.LR = self.dt*np.repeat(self.l[:,np.newaxis], self.M, axis=1) + np.repeat(self.r[np.newaxis,:], self.N, axis=0)
        self.Q  = self.dt*np.real(np.mean((np.exp(self.LR/2.) - 1.)/self.LR, 1))
        self.f1 = self.dt*np.real( np.mean( (-4. -    self.LR              + np.exp(self.LR)*( 4. - 3.*self.LR + self.LR**2) )/(self.LR**3) , 1) )
        self.f2 = self.dt*np.real( np.mean( ( 2. +    self.LR              + np.exp(self.LR)*(-2. +    self.LR             ) )/(self.LR**3) , 1) )
        self.f3 = self.dt*np.real( np.mean( (-4. - 3.*self.LR - self.LR**2 + np.exp(self.LR)*( 4. -    self.LR             ) )/(self.LR**3) , 1) )
        self.g  = -0.5j*self.nu*self.k

    def etdrk(self, Fforcing, u, v):

        # Computation is based on v = fft(u), so linear term is diagonal.
        # The time-discretization is done via ETDRK4
        # (exponential time differencing - 4th order Runge Kutta)

        Nv = self.g*fft(np.real(ifft(v))**2)
        a = self.E2*v + self.Q*Nv;
        Na = self.g*fft(np.real(ifft(a))**2)
        b = self.E2*v + self.Q*Na;
        Nb = self.g*fft(np.real(ifft(b))**2)
        c = self.E2*a + self.Q*(2.*Nb - Nv);
        Nc = self.g*fft(np.real(ifft(c))**2)

        v_ = self.E*v + (Nv + Fforcing)*self.f1 + 2.*(Na + Nb + Fforcing)*self.f2 + (Nc + Fforcing)*self.f3
        u_ = np.real(ifft(v_))

        return (u_, v_)

    def IC(self, u0=None, v0=None, case='noise', seed=42):

        # Set initial condition
        if (v0 is None):
            if (u0 is None):

                    # Gaussian noise (according to https://arxiv.org/pdf/1906.07672.pdf)
                    if case == 'noise':
                        #print("[KS] Noisy IC")
                        u0 = np.random.normal(0., 1e-3, self.N)
                    elif case == 'ETDRK4':
                        u0 = np.cos(self.x / 16)*(1+np.sin(self.x / 16))
                    else:
                        print("[KS] Error: IC case unknown")
                        return -1

            else:
                # check the input size
                if (np.size(u0,0) != self.N):
                    print("[KS] Error: wrong IC array size")
                    sys.exit()

                else:
                    # if ok cast to np.array
                    u0 = np.array(u0)

            # in any case, set v0:
            v0 = fft(u0)

        else:
            print("[KS] v0 was given")
            # the initial condition is provided in v0
            # check the input size
            if (np.size(v0,0) != self.N):
                print("[KS] Error: wrong IC array size")
                sys.exit()

            else:
                # if ok cast to np.array
                v0 = np.array(v0)
                # and transform to physical space
                u0 = np.real(ifft(v0))

        # and save to self
        self.u0  = u0
        self.u   = u0
        self.v0  = v0
        self.v   = v0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition

        # store the IC in [0]
        self.uu[0,:] = u0
        self.vv[0,:] = v0
        self.tt[0]   = 0.

    def setGroundTruth(self, t, x, uu):
        self.uu_truth = uu
        self.f_truth = interpolate.interp2d(x, t, self.uu_truth, kind='cubic')

    def mapGroundTruth(self):
        t = np.arange(0, self.uu.shape[0])*self.dt
        return self.f_truth(self.x,t)

    def step( self, Cs=None ):

        Fforcing = np.zeros(self.N, dtype=np.complex64)
        #
        dx = self.dx
        dx_ = 2*dx
        dx2 = dx*dx
        dx3 = dx2*dx
        dx4 = dx3*dx

        u = self.uu[self.ioutnum,:]
        um = np.roll(u, 1)
        umm = np.roll(u, 2)
        up = np.roll(u, -1)
        upp = np.roll(u, -2)

        dudx = (u - um)/dx
        d2udx2 = (up - 2*u + um)/dx
        d3udx3 = (upp - 2*up + 2*um - umm)/(2*dx3)
        d4udx4 = (upp - 4*up + 6*u - 4*um + umm)/dx4

        eps = np.finfo(float).eps

        if self.ssm == True:

            #sgs = 2*Cs*Cs*dx2*(d2udx2)*(dudx**2)/(np.absolute(dudx)+eps)
            #sgs = 2*Cs*Cs*dx2*(d4udx4*np.absolute(dudx) + d4udx4*dudx*d2udx2/(np.absolute(dudx)+eps))
            sgs = 2*Cs*Cs*dx2*np.absolute(dudx)*d2udx2
            #sgs = 2*Cs*Cs*dx2*np.absolute(dudx)*d4udx4
            Fforcing += fft( sgs )

            self.sgsHistory[self.ioutnum,:] = sgs

        if self.dsm == True:

            u_ = np.convolve(u, self.Gker)
            u2_ = np.convolve(u*u, self.Gker)
            L_ = u2_ - u_

            dudx_ = np.convolve(dudx, self.Gker)
            d3udx3_ = np.convolve(d3udx3, self.Gker)
            dudx_abs = np.absolute(dudx_)
            d3udx3_abs = np.absolute(d3udx3_)
            M = dx*dx*np.convolve(np.absolute(dudx)*dudx, self.Gker) - dx_*dx_*dudx_abs*dudx_
            #M = dx*dx*np.convolve(np.absolute(dudx)*d3udx3, self.Gker) - dx_*dx_*dudx_abs*d3udx3_

            C = np.mean(L_*M)/(2*np.mean(M*M))
            #print(C)
            sgs = 2*C*C*dx2*(d2udx2)*(dudx**2)/(np.absolute(dudx))
            #sgs = 2*C*C*dx2*(d4udx4*np.absolute(dudx) + d4udx4*dudx*d2udx2/(np.absolute(dudx)+eps))
            Fforcing += fft( sgs )

            self.sgsHistory[self.ioutnum,:] = sgs

        u, v = self.etdrk(Fforcing, self.u, self.v)

        self.stepnum += 1
        self.t       += self.dt

        self.u = u
        self.v = v

        self.ioutnum += 1
        self.uu[self.ioutnum,:] = u
        self.vv[self.ioutnum,:] = v
        self.tt[self.ioutnum]   = self.t


    def simulate(self, nsteps=None, restart=False, correction=[], Cs=None):
        #
        # If not provided explicitly, get internal values
        if (nsteps is None):
            nsteps = self.nsteps
        else:
            nsteps = int(nsteps)
            self.nsteps = nsteps

        if restart:
            # update nout in case nsteps or iout were changed
            nout      = nsteps
            self.nout = nout
            self.uut  = -1
            self.stepnum = 0
            self.ioutnum = 0
            # reset simulation arrays with possibly updated size
            self.__setup_timeseries(nout=self.nout)
            self.vv[0,:] = self.v0
            self.uu[0,:] = self.u0

        # advance in time for nsteps steps
        try:
            if (correction==[]):
                for n in range(1,self.nsteps+1):
                    self.step(Cs=Cs)
            else:
                for n in range(1,self.nsteps+1):
                    self.step(Cs=Cs)
                    self.v += correction

        except FloatingPointError:
            print("[KS] Floating point exception occured", flush=True)
            # something exploded
            # cut time series to last saved solution and return
            self.nout = self.ioutnum
            self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
            self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
            return -1

    def fou2real(self):
        # Convert from spectral to physical space
        if self.uut < self.stepnum:
            self.uut = self.stepnum
            self.uu = np.real(ifft(self.vv))

    def compute_Ek(self):
        #
        # compute all forms of kinetic energy
        #
        # Kinetic energy as a function of wavenumber and time
        self.__compute_Ek_kt()

        # Time-averaged energy spectrum as a function of wavenumber
        self.Ek_k = np.sum(self.Ek_kt, 0)/(self.ioutnum+1) # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0]

        # Total kinetic energy as a function of time
        self.Ek_t = np.sum(self.Ek_kt, 1)

        # Time-cumulative average as a function of wavenumber and time
        self.Ek_ktt = np.cumsum(self.Ek_kt, 0)[:self.ioutnum+1,:] / np.arange(1,self.ioutnum+2)[:,None] # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero

        # Time-cumulative average as a function of time
        self.Ek_tt = np.cumsum(self.Ek_t, 0)[:self.ioutnum+1] / np.arange(1,self.ioutnum+2) # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero

    def __compute_Ek_kt(self):
        try:
            self.Ek_kt = 1./2.*np.real( self.vv.conj()*self.vv / self.N ) * self.dx
        except FloatingPointError:
            #
            # probable overflow because the simulation exploded, try removing the last solution
            problem=True
            remove=1
            self.Ek_kt = np.zeros([self.nout+1, self.N]) + 1e-313
            while problem:
                try:
                    self.Ek_kt[0:self.nout+1-remove,:] = 1./2.*np.real( self.vv[0:self.nout+1-remove].conj()*self.vv[0:self.nout+1-remove] / self.N ) * self.dx
                    problem=False
                except FloatingPointError:
                    remove+=1
                    problem=True
        return self.Ek_kt

    def getReward(self):
        # Convert from spectral to physical space
        self.fou2real()

        u = self.uu[self.ioutnum,:]
        t = [self.t]
        uMap = self.f_truth(self.x, t)
        return -np.abs(u-uMap)

    def getState(self):
        # Convert from spectral to physical space
        self.fou2real()

        # Extract state
        u = self.uu[self.ioutnum,:]

        up = np.roll(u,-1)
        um = np.roll(u,+1)
        dudx = (up - um)/(2.*self.dx)
        d2udx2 = (up - 2.*u + um)/self.dx**2

        state = np.concatenate((dudx, d2udx2))

        return state

    def compute_Sgs(self, nURG):
        hidx = np.abs(self.k)>nURG//2
        self.sgsHistory = np.zeros(self.uu.shape)

        for idx in range(self.uu.shape[0]):
            # calc uhat(t)
            u = self.uu[idx,:]
            u2 = u*u
            v = fft(u)
            v2 = fft(u2)
            vh = v
            v2h = v2
            vh[hidx] = 0
            v2h[hidx] = 0

            uh = np.real(ifft(vh))
            uhm = np.roll(uh,+1)
            u2h = np.real(ifft(v2h))
            u2hm = np.roll(u2h,+1)

            # calc latteral derivatives
            duhdx = (uh - uhm)/self.dx
            du2hdx = (u2h - u2hm)/self.dx

            self.sgsHistory[idx,:] = -uh*duhdx  + 0.5*du2hdx
