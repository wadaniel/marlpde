import sys
import time
from numpy import pi
from scipy import interpolate
from scipy.fftpack import rfft, irfft, rfftfreq
import numpy as np

np.seterr(over='raise', invalid='raise')
def gaussian( x, mean, sigma ):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-1/2*( (x-mean)/sigma )**2)

def hat( x, mean, dx ):
    left  = np.clip((x + dx - mean)/dx, a_min = 0., a_max = 1.)
    right = np.clip((dx - x + mean)/dx, a_min = 0., a_max = 1.)
    return left + right - 1.

class Diffusion:  
    #
    # Solution to the Diffusion equation
    #
    # u_t = nu*u_xx
    # with periodic BCs on x \in [0, L]: u(0,t) = u(L,t).

    def __init__(self, L=2.*np.pi, N=512, dt=0.001, nu=0.0, dforce=True, nsteps=None, tend=5., u0=None, v0=None, case=None, ssm=False, dsm=False, noise=0., seed=42, version=0, nunoise=False):
        
        # Randomness
        np.random.seed(None)
        self.noise = noise*L
        self.seed = seed

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
 
        if (2.*self.nu*self.dt >= self.dx**2):
            print("[Diffusion] Warning: CFL condition violated", flush=True)

        # Basis
        self.M = 0
        self.basis = None
        self.actions = None
        self.dforce = dforce
        self.version = version
        if (self.version > 1):
            print("[Diffusion] Version not recognized", flush=True)
            sys.exit()

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
        elif (u0 is not None):
            self.IC(u0 = u0)
        else:
            print("[Diffusion] IC ambigous")
            sys.exit()
        
    def __setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        
        # nout+1 because we store the IC as well
        self.uu = np.zeros([self.nout+1, self.N])
        self.tt = np.zeros(self.nout+1)
        self.sgsHistory = np.zeros([self.nout+1, self.N])
        self.actionHistory = np.zeros([self.nout+1, self.N])
        
        self.tt[0]   = 0.

    def setup_basis(self, M, kind = 'uniform'):
        self.M = M
        
        # Action record
        if M > 1:
            if kind == 'uniform':
                self.basis = np.zeros((self.M, self.N))
                for i in range(self.M):
                    assert self.N % self.M == 0, print("[Diffusion] Something went wrong in basis setup")
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
                print("[Diffusion] Basis function not known, exit..")
                sys.exit()
        else:
            self.basis = np.ones((self.M, self.N))
        
        np.testing.assert_allclose(np.sum(self.basis, axis=0), 1)

    def IC(self, u0=None, case='box'):
        
        # Set initial condition
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
                
                rng = 123456789 + self.seed
                a = 1103515245
                c = 12345
                m = 2**13
            
                A = 1
                u0 = np.ones(self.N)
                for k in range(1, self.N):
                    offset = np.random.normal(loc=0., scale=self.noise) if self.noise > 0 else 0.
                    rng = (a * rng + c) % m
                    phase = rng/m*2.*np.pi

                    Ek = A*5**(-5/3) if k <= 5 else A*k**(-5/3) 
                    u0 += np.sqrt(2*Ek)*np.sin(k*2*np.pi*self.x/self.L + phase + offset)
                    
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

            elif case == 'zero':
                u0 = np.zeros(self.N)
            
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
                print("[Diffusion] Error: IC case unknown")
                sys.exit()

        else:
            # check the input size
            if (np.size(u0,0) != self.N):
                print("[Diffusion] Error: wrong IC array size (is {}, expected {}".format(np.size(u0,0),self.N))
                sys.exit()

            else:
                # if ok cast to np.array
                u0 = np.array(u0)
        
        # and save to self
        self.u0  = u0
        self.v0  = rfft(u0)
        self.u   = u0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition
  
        # store the IC in [0]
        self.uu[0,:] = u0
        self.tt[0]   = 0.
       
    def setGroundTruth(self, t, x, uu):
        self.uu_truth = uu
        self.f_truth = interpolate.interp2d(x, t, self.uu_truth, kind='cubic')
 
    def mapGroundTruth(self):
        t = np.arange(0,self.uu.shape[0])*self.dt
        return self.f_truth(self.x,t)

    def getAnalyticalSolution(self, t):
        if t > 0.:
            print("[Diffusion] TODO: Analytical solution")
            sys.exit()

        else:
            sol = self.u0

        return sol
 
    def step( self, actions=None ):
        
        forcing = np.zeros(self.N)

        if (actions is not None):
            assert self.basis is not None, print("[Diffusion] Basis not set up (is None).")
            assert len(actions) == self.M, print("[Diffusion] Wrong number of actions (provided {}/{}".format(len(actions), self.M))

            forcing = np.matmul(actions, self.basis)
            self.actionHistory[self.ioutnum,:] = forcing
            
            if self.dforce == False:
                u = self.uu[self.ioutnum,:]
                up = np.roll(u,1)
                up[-1] = 0.
                um = np.roll(u,-1)
                um[0] = 0.
                d2udx2 = (up - 2.*u + um)/self.dx**2
                forcing *= d2udx2
            
            self.sgsHistory[self.ioutnum,:] = forcing

        """
        Expl. Euler Central Differences
        """
        up = np.roll(self.u, -1)
        up[-1] = 0.
        um = np.roll(self.u, +1)
        um[0] = 0.
        d2udx2 = (up - 2.*self.u + um)/self.dx**2

        self.u = self.u + self.dt * (self.nu * d2udx2 + forcing)
     
        self.stepnum += 1
        self.t       += self.dt
 
        self.ioutnum += 1
        self.uu[self.ioutnum,:] = self.u
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
            self.uu[0,:] = self.u0
 
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
            print("[Burger] Floating point exception occured in simulate", flush=True)
            # something exploded
            # cut time series to last saved solution and return
            self.nout = self.ioutnum
            self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
            return -1

    def getMseReward(self):
        try:
            uTruthToCoarse = self.mapGroundTruth()
            uDiffMse = ((uTruthToCoarse[self.ioutnum,:] - self.uu[self.ioutnum,:])**2).mean()
        except FloatingPointError:
            print("[Burger] Floating point exception occured in mse", flush=True)
            return -np.inf

        return -uDiffMse
     
    def getState(self, nAgents = None):
        try:
            # Extract state
            u = self.uu[self.ioutnum,:]
            umt = self.uu[self.ioutnum-1,:] if self.ioutnum > 0 else self.uu[self.ioutnum, :]

            dudt = (u - umt)/self.dt
                 
            up = np.roll(u,1)
            um = np.roll(u,-1)
            d2udx2 = (up - 2.*u + um)/self.dx**2
         
            if self.version == 0:
                state = d2udx2
            elif self.version == 1:
                state = np.concatenate((dudt,d2udx2))

        except FloatingPointError:

            print("[Diffusion] Floating point exception occured in getState", flush=True)
            if self.version == 0:
                return np.inf*np.ones(self.N)
            elif self.version == 1:
                return np.inf*np.ones(2*self.N)
       
        return state
