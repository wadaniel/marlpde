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

class Burger:
    #
    # Solution of the Burgers equation
    #
    # u_t + u*u_x = nu*u_xx0
    # with periodic BCs on x \in [0, L]: u(0,t) = u(L,t).

    def __init__(self, L=2.*np.pi, N=512, dt=0.001, nu=0.0, nsteps=None, tend=150, u0=None, v0=None, case=None, noise=0., seed=42):
        
        # Randomness
        self.noise = noise*L
        self.seed = seed
       
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
        
        # save to self
        self.L      = L
        self.N      = N
        self.dx     = L/N
        #self.x      = np.linspace(-0.5*self.L, +0.5*self.L, N, endpoint=False)
        self.x      = np.linspace(0, self.L, N, endpoint=False)
        self.dt     = dt
        self.nu     = nu
        self.nsteps = nsteps
        self.nout   = nsteps
 
        # Basis
        self.M = 0
        self.basis = None

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
            print("[Burger] IC ambigous")
            sys.exit()
        
        # precompute Fourier-related quantities
        self.__setup_fourier()
        
    def __setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        
        # nout+1 because we store the IC as well
        self.uu = np.zeros([self.nout+1, self.N])
        self.vv = np.zeros([self.nout+1, self.N], dtype=np.complex64)
        self.tt = np.zeros(self.nout+1)
        
        self.tt[0]   = 0.

    def __setup_fourier(self, coeffs=None):
        #self.x  = 2*pi*self.L*np.r_[0:self.N]/self.N
        #self.k  = np.r_[0:self.N/2, 0, -self.N/2+1:0]/self.L # Wave numbers
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

    def setup_basis(self, M, kind = 'uniform'):
        self.M = M
        if M > 1:
            if kind == 'uniform':
                self.basis = np.zeros((self.M, self.N))
                for i in range(self.M):
                    assert self.N % self.M == 0, print("[Burger] Something went wrong in basis setup")
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
                print("[Burger] Basis function not known, exit..")
                sys.exit()
        else:
            self.basis = np.ones((self.M, self.N))
        
        np.testing.assert_allclose(np.sum(self.basis, axis=0), 1)

    def IC(self, u0=None, v0=None, case='box'):
        
        # Set initial condition
        if (v0 is None):
            if (u0 is None):
                    
                    offset = np.random.normal(loc=0., scale=self.noise) if self.noise > 0 else 0.
                    
                    # Gaussian initialization
                    if case == 'gaussian':
                        # Gaussian noise (according to https://arxiv.org/pdf/1906.07672.pdf)
                        #u0 = np.random.normal(0., 1, self.N)
                        sigma = self.L/8
                        u0 = gaussian(self.x, mean=0.5*self.L+offset, sigma=sigma)
                        
                    # Box initialization
                    elif case == 'box':
                        u0 = np.abs(self.x-self.L/2-offset)<self.L/8
                    
                    # Sinus
                    elif case == 'sinus':
                        u0 = np.sin(self.x+offset)
 
                    # Turbulence
                    elif case == 'turbulence':
                        # Taken from: 
                        # A priori and a posteriori evaluations 
                        # of sub-grid scale models for the Burgers' eq. (Li, Wang, 2016)
                        
                        phase = 123456789 + self.seed
                        a = 1103515245
                        c = 12345
                        m = 2**13
                    
                        A = 1
                        u0 = np.ones(self.N)
                        for k in range(1, self.N):
                            #offset = np.random.normal(loc=0., scale=self.noise) if self.noise > 0 else 0.
                            offset = np.random.uniform(low=0., high=2*np.pi) if self.noise > 0 else 0.
                            phase = (a * phase + c) % m 
                            Ek = A*5**(-5/3) if k <= 5 else A*k**(-5/3) 
                            u0 += np.sqrt(2*Ek)*np.sin(k*2*np.pi*self.x/self.L+phase + offset)

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

                    else:
                        print("[Burger] Error: IC case unknown")
                        sys.exit()

            else:
                # check the input size
                if (np.size(u0,0) != self.N):
                    print("[Burger] Error: wrong IC array size")
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
                print("[Burger] Error: wrong IC array size")
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
        t = np.arange(0,self.uu.shape[0])*self.dt
        return self.f_truth(self.x,t)

    def getAnalyticalSolution(self, t):
        print("[Burger] TODO.. exit")
        sys.exit()
 
    def step( self, actions=None ):

        Fforcing = np.zeros(self.N)

        if (actions is not None):
            assert self.basis is not None, print("[Burger] Basis not set up (is None).")
            assert len(actions) == self.M, print("[Burger] Wrong number of actions (provided {}/{}".format(len(actions), self.M))
            forcing = np.matmul(actions, self.basis)

            u = self.uu[self.ioutnum,:]

            up = np.roll(u,1)
            um = np.roll(u,-1)
            d2udx2 = (up - 2.*u + um)/self.dx**2

            Fforcing = fft( forcing*d2udx2 )

            self.v = self.v - self.dt*0.5*self.k1*fft(self.u**2) + self.dt*self.nu*self.k2*self.v + self.dt*Fforcing 
        
        else:
            self.v = self.v - self.dt*0.5*self.k1*fft(self.u**2) + self.dt*self.nu*self.k2*self.v

        # Impl-expl step (TODO)
        #self.v = (self.v - self.dt*0.5*self.k1*fft(self.u**2) + self.dt*Fforcing) / (1. - self.dt*self.nu*self.k2*self.v)
        
        self.u = np.real(ifft(self.v))
        
        self.stepnum += 1
        self.t       += self.dt
 
        self.ioutnum += 1
        self.uu[self.ioutnum,:] = self.u
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
            print("[Burger] Floating point exception occured", flush=True)
            # something exploded
            # cut time series to last saved solution and return
            self.nout = self.ioutnum
            self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
            self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
            return -1

    def fou2real(self):
        # Convert from spectral to physical space
        self.uut = self.stepnum
        if self.stepnum < self.uut:
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

    def getMseReward(self):
        uTruthToCoarse = self.mapGroundTruth()
        #print(uTruthToCoarse[self.ioutnum,:])
        #print(self.uu[self.ioutnum,:])
        uDiffMse = ((uTruthToCoarse[self.ioutnum,:] - self.uu[self.ioutnum,:])**2).mean()
        return -uDiffMse
     
    def getState(self, nAgents = None):
        # Convert from spectral to physical space
        #self.iou2real()

        # Extract state
        u = self.uu[self.ioutnum,:]
        
        #dudu = np.zeros(self.N)
        #dudu[:-1] = (u[1:]-u[:-1])/self.dx
        #dudu[-1] = dudu[-2]
        #dudt = (self.uu[self.ioutnum,:]-self.uu[self.ioutnum-1,:])/self.dt
        #state = np.column_stack( (u, dudu, dudt) )
        #state = np.column_stack( (u, dudt) )
        #state = u
             
        up = np.roll(u,1)
        um = np.roll(u,-1)
        d2udx2 = (up - 2.*u + um)/self.dx**2
        
        state = d2udx2
       
        return state
