import sys
from numpy import pi
from scipy import interpolate
from scipy.fftpack import fft, ifft
import numpy as np

np.seterr(over='raise', invalid='raise')

def gaussian( x, mean, sigma ):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-1/2*( (x-mean)/sigma )**2)

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

    def __init__(self, L=16, N=128, dt=0.25, nu=1.0, nsteps=None, tend=150, u0=None, v0=None, case=None, coeffs=None, noisy = False):
        
        # Initialize
        L  = float(L); 
        dt = float(dt); 
        tend = float(tend)
        
        if (nsteps is None):
            nsteps = int(tend/dt)
        else:
            nsteps = int(nsteps)
            # override tend
            tend = dt*nsteps
        
        self.noisy = noisy

        # save to self
        self.L      = L
        self.N      = N
        self.dx     = 2*pi*L/N
        self.x      = 2*pi*self.L*np.r_[0:self.N]/self.N
        self.dt     = dt
        self.nu     = nu
        self.nsteps = nsteps
        self.nout   = nsteps
        self.sigma  = 2*pi*L/(2*N)
        self.coeffs = coeffs
        
        # time when field space transformed
        self.uut = -1
        # field in real space
        self.uu = None
        # ground truth in real space
        self.uu_truth = None
        # interpolation of truth
        self.f_truth = None

        # set initial condition
        if (u0 is None) and (v0 is None):
            self.IC()
        elif (u0 is not None):
            self.IC(u0 = u0)
        elif (v0 is not None):
            self.IC(v0 = v0)
        else:
            print("[KS] IC ambigous")
            sys.exit()
 
        # initialize simulation arrays
        self.__setup_timeseries()
        
        # precompute Fourier-related quantities
        self.__setup_fourier(self.coeffs)
        
        # precompute ETDRK4 scalar quantities:
        self.__setup_etdrk4()

        # Gaussians for forcing terms
        self.__setup_gaussians()


    def __setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        
        # nout+1 because we store the IC as well
        self.vv = np.zeros([self.nout+1, self.N], dtype=np.complex64)
        self.tt = np.zeros(self.nout+1)
        
        # store the IC in [0]
        self.vv[0,:] = self.v0
        self.tt[0]   = 0.

    def __setup_fourier(self, coeffs=None):
        self.k  = np.r_[0:self.N/2, 0, -self.N/2+1:0]/self.L # Wave numbers
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
        self.M  = 62                                           # no. of points for complex means
        self.r  = np.exp(1j*pi*(np.r_[1:self.M+1]-0.5)/self.M) # roots of unity
        self.LR = self.dt*np.repeat(self.l[:,np.newaxis], self.M, axis=1) + np.repeat(self.r[np.newaxis,:], self.N, axis=0)
        self.Q  = self.dt*np.real(np.mean((np.exp(self.LR/2.) - 1.)/self.LR, 1))
        self.f1 = self.dt*np.real( np.mean( (-4. -    self.LR              + np.exp(self.LR)*( 4. - 3.*self.LR + self.LR**2) )/(self.LR**3) , 1) )
        self.f2 = self.dt*np.real( np.mean( ( 2. +    self.LR              + np.exp(self.LR)*(-2. +    self.LR             ) )/(self.LR**3) , 1) )
        self.f3 = self.dt*np.real( np.mean( (-4. - 3.*self.LR - self.LR**2 + np.exp(self.LR)*( 4. -    self.LR             ) )/(self.LR**3) , 1) )
        self.g  = -0.5j*self.k
 
    def __setup_gaussians(self):
        self.gaussians = np.zeros(self.N)
        for i in range(self.N):
            mean = i*self.L/4
            self.gaussians[i] = gaussian( mean, mean, self.sigma)

    def IC(self, u0=None, v0=None, seed=42):
        
        # Set initial condition
        if (v0 is None):
            if (u0 is None):
                    if self.noisy:
                        print("[KS] Using Gaussian initial condition...")
                    
                    # uniform noise
                    # Gaussian noise (according to https://arxiv.org/pdf/1906.07672.pdf)
                    np.random.seed( seed )
                    u0 = np.random.normal(0., 1e-4, self.N)
                    
                    # Gaussian initialization
                    #sigma = 1/(2*np.pi)
                    #u0 = np.exp(-0.5/(sigma*sigma)*(np.linspace(0, self.L, self.N) - 0.5*self.L)**2)*1/np.sqrt(2*np.pi*sigma*sigma)
            else:
                # check the input size
                if (np.size(u0,0) != self.N):
                    if self.noisy:
                        print("[KS] Error: wrong IC array size")
                    return -1
                else:
                    if self.noisy:
                        print("[KS] Using given (real) flow field...")
                    # if ok cast to np.array
                    u0 = np.array(u0)
            # in any case, set v0:
            v0 = fft(u0)
        else:
            # the initial condition is provided in v0
            # check the input size
            if (np.size(v0,0) != self.N):
                if self.noisy:
                    print("[KS] Error: wrong IC array size")
                    return -1
            else:
                if self.noisy:
                    print("[KS] Using given (Fourier) flow field...")
                # if ok cast to np.array
                v0 = np.array(v0)
                # and transform to physical space
                u0 = ifft(v0)
        #
        # and save to self
        self.u0  = u0
        self.v0  = v0
        self.v   = v0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition
        
    def setGroundTruth(self, t, x, uu):
        self.uu_truth = uu
        self.f_truth = interpolate.interp2d(x, t, self.uu_truth, kind='cubic')
 
    def mapGroundTruth(self):
        t = np.arange(0, self.uu.shape[0])*self.dt
        return self.f_truth(self.x,t)


    def step( self, action=None ):
        forcing  = np.zeros(self.N)
        Fforcing = np.zeros(self.N)
 
        if (action is not None):
            if len(action) > 1:
                assert len(action) == self.nActions, print("Wrong number of actions. provided {}/{}".format(len(action), self.nActions))
            forcing += action*self.gaussians[:]
            Fforcing = fft( forcing )


        #
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
        
        if (action is not None):
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
            # reset simulation arrays with possibly updated size
            self.__setup_timeseries(nout=self.nout)
        
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
        #self.uut = self.stepnum
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

    def space_filter(self, k_cut=2):
        #
        # spatially filter the time series
        self.uu_filt  = np.zeros([self.nout+1, self.N])
        for n in range(self.nout+1):
            v_filt = np.copy(self.vv[n,:])    # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[np.abs(self.k)>=k_cut] = 0 # set to zero wavenumbers > k_cut
            self.uu_filt[n,:] = np.real(ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt

    def space_filter_int(self, k_cut=2, N_int=10):
        #
        # spatially filter the time series
        self.N_int        = N_int
        self.uu_filt      = np.zeros([self.nout+1, self.N])
        self.uu_filt_int  = np.zeros([self.nout+1, self.N_int])
        self.x_int        = 2*pi*self.L*np.r_[0:self.N_int]/self.N_int
        for n in range(self.nout+1):
            v_filt = np.copy(self.vv[n,:])   # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[np.abs(self.k)>=k_cut] = 313e6
            v_filt_int = v_filt[v_filt != 313e6] * self.N_int/self.N
            self.uu_filt_int[n,:] = np.real(ifft(v_filt_int))
            v_filt[np.abs(self.k)>=k_cut] = 0
            self.uu_filt[n,:] = np.real(ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt

    def getReward(self):
        # Convert from spectral to physical space
        self.fou2real()
        
        u = self.uu[self.ioutnum,:]
        t = [self.t]
        uMap = self.f_truth(self.x, t)
        return -np.abs(u-uMap)

    def getState(self, nAgents = None):
        # Convert from spectral to physical space
        self.fou2real()

        # Extract state
        u = self.uu[self.ioutnum,:]
        dudu = np.zeros(self.N)
        dudu[:-1] = (u[1:]-u[:-1])/self.dx
        dudu[-1] = dudu[-2]
        dudt = (self.uu[self.ioutnum,:]-self.uu[self.ioutnum-1,:])/self.dt
        state = np.column_stack( (u, dudu, dudt) )
        return state

    def updateField(self, factors):
        # elementwise multiplication
        self.v = self.v * factors
