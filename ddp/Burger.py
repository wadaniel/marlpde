import sys
import time
from numpy import pi
from scipy import interpolate
from scipy.fftpack import fft, ifft, fftfreq
import numpy as np

class Burger:
    #
    # Solution of the Burgers equation
    #
    # u_t + u*u_x = nu*u_xx0 + Forcing
    # with periodic BCs on x \in [0, L]: u(0,t) = u(L,t).

    def __init__(self, 
            L=2.*np.pi, 
            N=512, 
            dt=0.001, 
            nu=0.0, 
            nsteps=None, 
            tend=5., 
            u0=None, 
            v0=None, 
            case=None, 
            forcing=False, 
            noise=0., 
            seed=42, 
            fseed=42,
            nunoise=False, 
            stepper=1):
        
        # Randomness
        np.random.seed(None)
        self.noise = noise*L
        self.offset = np.random.normal(loc=0., scale=self.noise) if self.noise > 0. else 0.
        while np.abs(self.offset) > L:
            self.offset = np.random.normal(loc=0., scale=self.noise) if self.noise > 0. else 0.

        self.stepper = stepper
        
        # seed of turbulent IC and forcing
        self.tseed = seed
        np.random.seed(fseed)

        # Apply forcing term?
        self.forcing = forcing

        # Initialize
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
        self.nu     = nu
        if nunoise:
            self.nu = 0.01+0.02*np.random.uniform()
        self.nsteps = nsteps
        self.nout   = nsteps
 
        # random factors for forcing
        self.randfac1 = np.random.normal(loc=0., scale=1., size=(32,nsteps)) # scale
        self.randfac2 = np.random.normal(loc=0., scale=1., size=(32,nsteps)) # phase
        
        # field in real space
        self.uu = None
        # temporal quantity for ABCN scheme
        self.Fn_old = None
 
        # initialize simulation arrays
        self.__setup_timeseries()
  
        # precompute Fourier-related quantities
        self.__setup_fourier()
 
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
            print("[Burger] IC ambigous")
            sys.exit()
       
    def __setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        
        # nout+1 because we store the IC as well
        self.uu = np.zeros([self.nout+1, self.N])
        self.vv = np.zeros([self.nout+1, self.N], dtype=np.complex64)
        self.tt = np.zeros(self.nout+1)
        self.f  = np.zeros([self.nout+1, self.N])
        self.sgsHistory = np.zeros([self.nout+1, self.N])
        self.actionHistory = np.zeros([self.nout+1, self.N])
        
        self.tt[0]   = 0.

    def __setup_fourier(self, coeffs=None):
        self.k   = fftfreq(self.N, self.L / (2*np.pi*self.N))
        self.k1  = 1j * self.k
        self.k2  = self.k1**2
 
        # Fourier multipliers for the linear term Lu
        if (coeffs is None):
            # normal-form equation
            self.l = self.nu*self.k**2
        else:
            # altered-coefficients 
            self.l = -      coeffs[0]*np.ones(self.k.shape) \
                     -      coeffs[1]*1j*self.k             \
                     + (1 + coeffs[2])  *self.k**2          \
                     +      coeffs[3]*1j*self.k**3          \
                     - (1 + coeffs[4])  *self.k**4

    def IC(self, u0=None, v0=None, case='zero'):
        
        # Set initial condition
        if (v0 is None):
            if (u0 is None):
                    
                    # Sinus
                    if case == 'sinus':
                        u0 = np.sin(4.*np.pi*(self.x+self.offset)/self.L)
                    
                    # Turbulence
                    elif case == 'turbulence':
                        # Taken from: 
                        # A priori and a posteriori evaluations 
                        # of sub-grid scale models for the Burgers' eq. (Li, Wang, 2016)
                        
                        rng = 123456789 + self.tseed
                        a = 1103515245
                        c = 12345
                        m = 2**13
                    
                        A = 1
                        u0 = np.ones(self.N)
                        for k in range(1, self.N):
                            rng = (a * rng + c) % m
                            phase = rng/m*2.*np.pi

                            Ek = A*5**(-5/3) if k <= 5 else A*k**(-5/3) 
                            u0 += np.sqrt(2*Ek)*np.sin(k*2*np.pi*(self.x+self.offset)/self.L + phase)
                            
                        # rescale IC
                        idx = 0
                        criterion = np.sqrt(np.sum((u0-1.)**2)/self.N)
                        while (criterion < 0.65 or criterion > 0.75):
                            scale = 0.7/criterion
                            u0 *= scale
                            criterion = np.sqrt(np.sum((u0-1.)**2)/self.N)
                            
                            # exit
                            idx += 1
                            if idx > 100:
                                break
                        
                        assert( criterion < 0.8 )
                        assert( criterion > 0.6 )

                    elif case == 'forced':
                        u0 = np.zeros(self.N)
                        #A = 1
                        A = 1./self.N
                        for k in range(1,self.N):
                            r1 = np.random.normal(loc=0., scale=1.)
                            r2 = np.random.normal(loc=0., scale=1.)
                            #A = A**(-5/3) if k <= 5 else A*k**(-5/3) 
                            u0 += r1*A*np.sin(2.*np.pi*(k*self.x/self.L+r2))
 
                    else:
                        print("[Burger] Error: IC case unknown")
                        sys.exit()

            else:
                # check the input size
                if (np.size(u0,0) != self.N):
                    print("[Burger] Error: wrong IC array size (is {}, expected {}".format(np.size(u0,0),self.N))
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
                print("[Burger] Error: wrong IC array size (is {}, expected {}".format(np.size(v0,0),self.N))
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
        
        # init temporal quantity for ABCN
        self.Fn_old =  self.k1*fft(0.5*self.u**2) 
       
    def step( self, actions=None ):

        if self.forcing:
            # Compute forcing terms
            forcing = np.zeros(self.N)
         
            A=np.sqrt(2.)/self.L
            for k in range(1,4):
                ridx = self.ioutnum % self.stepper
                r1 = self.randfac1[k, ridx]
                r2 = self.randfac2[k, ridx] 
                forcing += r1*A/np.sqrt(k*self.stepper*self.dt)*np.cos(2*np.pi*k*(self.x+self.offset)/self.L+2*np.pi*r2);
        
            self.f[self.ioutnum, :] = forcing
        
        Fforcing = fft( self.f[self.ioutnum, :])
            
        if (actions is not None):
            self.actionHistory[self.ioutnum,:] = actions
            Fforcing += fft( actions )

        """
        Adam Bashfort / CN
        """
        C  = -0.5*self.k2*self.nu*self.dt
        Fn = self.k1*fft(0.5*self.u**2)
        self.v =((1.0-C)*self.v-0.5*self.dt*(3.0*Fn-self.Fn_old)+self.dt*Fforcing)/(1.0+C)
        self.Fn_old = Fn.copy()
 
        self.u = np.real(ifft(self.v))
        
        self.stepnum += 1
        self.t       += self.dt
 
        self.ioutnum += 1
        self.uu[self.ioutnum,:] = self.u
        self.vv[self.ioutnum,:] = self.v
        self.tt[self.ioutnum]   = self.t

    def simulate(self, nsteps=None):
        #
        # If not provided explicitly, get internal values
        if (nsteps is None):
            nsteps = self.nsteps
        else:
            nsteps = int(nsteps)
            self.nsteps = nsteps
        
        # advance in time for nsteps steps
        try:
            for n in range(1,self.nsteps+1):
                self.step()
                
        except FloatingPointError:
            print("[Burger] Floating point exception occured in simulate", flush=True)
            # something exploded
            # cut time series to last saved solution and return
            self.nout = self.ioutnum
            self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
            self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
            return -1

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
