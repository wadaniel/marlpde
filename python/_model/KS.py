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

class KS:
    #
    # Solution of the  KS equation
    #
    # u_t + u_xx + u_xxxx + 0.5u_x*u_x= 0,
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

    def __init__(self, L=2.*np.pi, N=512, dt=0.001, nu=1.0, dforce=True, nsteps=None, tend=5., u0=None, v0=None, case=None, noise=0., seed=42):
        
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
  
        # Basis
        self.M = 0
        self.basis = None
        self.actions = None

        self.dforce = dforce
       
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
        self.actionHistory = np.zeros([self.nout+1, self.N])
        
        self.tt[0]   = 0.

    def __setup_fourier(self, coeffs=None):
        self.k   = fftfreq(self.N, self.L / (2*np.pi*self.N))
        # Fourier multipliers for the linear term Lu
        if (coeffs is None):
            # normal-form equation
            self.l = self.k**2 - self.k**4 #(KS)
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
        self.MM = 62                                             # no. of points for complex means
        self.r  = np.exp(1j*pi*(np.r_[1:self.MM+1]-0.5)/self.MM) # roots of unity
        self.LR = self.dt*np.repeat(self.l[:,np.newaxis], self.MM, axis=1) + np.repeat(self.r[np.newaxis,:], self.N, axis=0)
        self.Q  = self.dt*np.real(np.mean((np.exp(self.LR/2.) - 1.)/self.LR, 1))
        self.f1 = self.dt*np.real( np.mean( (-4. -    self.LR              + np.exp(self.LR)*( 4. - 3.*self.LR + self.LR**2) )/(self.LR**3) , 1) )
        self.f2 = self.dt*np.real( np.mean( ( 2. +    self.LR              + np.exp(self.LR)*(-2. +    self.LR             ) )/(self.LR**3) , 1) )
        self.f3 = self.dt*np.real( np.mean( (-4. - 3.*self.LR - self.LR**2 + np.exp(self.LR)*( 4. -    self.LR             ) )/(self.LR**3) , 1) )
        self.g  = -0.5j*self.k
 
    def setup_basis(self, M, kind = 'uniform'):
        self.M = M
        if M > 1:

            if kind == 'uniform':
                self.basis = np.zeros((self.M, self.N))
                for i in range(self.M):
                    assert self.N % self.M == 0, print("[KS] Something went wrong in basis setup")
                    idx1 = i * self.N//self.M
                    idx2 = (i+1) * self.N//self.M
                    self.basis[i,idx1:idx2] = 1.

            elif kind == 'hat':
                self.basis = np.ones((self.M, self.N))
                dx = self.L/(self.M-1)
                for i in range(self.M):
                    mean = i*dx
                    self.basis[i,:] = hat( self.x, mean, dx )

            else:
                print("[KS] Basis function not known, exit..")
                sys.exit()
        else:
            self.basis = np.ones((self.M, self.N))
        
        np.testing.assert_allclose(np.sum(self.basis, axis=0), 1)

    def IC(self, u0=None, v0=None, case='noise', seed=42):
        
        # Set initial condition
        if (v0 is None):
            if (u0 is None):
                    
                    # Gaussian noise (according to https://arxiv.org/pdf/1906.07672.pdf)
                    if case == 'noise':
                        #print("[KS] Noisy IC")
                        u0 = np.random.normal(0., 1e-3, self.N)
                    
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


    def step( self, actions=None ):
 
        Fforcing = np.zeros(self.N)
        if (actions is not None):
            assert self.basis is not None, print("[KS] Basis not set up (is None).")
            assert len(actions) == self.M, print("[KS] Wrong number of actions (provided {}/ expected {})".format(len(actions), self.M))

            forcing = np.matmul(actions, self.basis)
            self.actionHistory[self.ioutnum,:] = forcing
            
            if self.dforce == False:
                u = self.uu[self.ioutnum,:]
                up = np.roll(u,1)
                um = np.roll(u,-1)
                d2udx2 = (up - 2.*u + um)/self.dx**2
                forcing *= d2udx2
                
            self.sgsHistory[self.ioutnum,:] = forcing
            
            Fforcing = fft( forcing )

        # Computation is based on v = fft(u), so linear term is diagonal.
        # The time-discretization is done via ETDRK4
        # (exponential time differencing - 4th order Runge Kutta)
        #
        v = self.v;                           
        Nv = self.g*fft(np.real(ifft(v))**2)
        a = self.E2*v + self.Q*Nv;            
        Na = self.g*fft(np.real(ifft(a))**2)
        b = self.E2*v + self.Q*Na;            
        Nb = self.g*fft(np.real(ifft(b))**2)
        c = self.E2*a + self.Q*(2.*Nb - Nv);  
        Nc = self.g*fft(np.real(ifft(c))**2)
        
        if (actions is not None):
            self.v = self.E*v + (Nv + Fforcing)*self.f1 + 2.*(Na + Nb + 2*Fforcing)*self.f2 + (Nc + Fforcing)*self.f3
        else:
            self.v = self.E*v + Nv*self.f1 + 2.*(Na + Nb)*self.f2 + Nc*self.f3

        self.stepnum += 1
        self.t       += self.dt
 
        self.ioutnum += 1
        self.vv[self.ioutnum,:] = self.v
        self.tt[self.ioutnum]   = self.t

    def simulate(self, nsteps=None, restart=False, correction=[]):
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
            self.uu[0,:] = self.v0
        
        # advance in time for nsteps steps
        try:
            if (correction==[]):
                for n in range(1,self.nsteps+1):
                    self.step()
            else:
                for n in range(1,self.nsteps+1):
                    self.step()
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
            print("[KS] Floating point exception occured", flush=True)
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
        self.sgsHistory = np.zeros(self.uu.shape)
        self.sgsHistoryAlt = np.zeros(self.uu.shape)
        self.sgsHistoryAlt2 = np.zeros((self.stepnum+1, nURG))
        # TODO
