import sys
import time
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
    # u_t + u*u_x = nu*u_xx0 + Forcing
    # with periodic BCs on x \in [0, L]: u(0,t) = u(L,t).

    def __init__(self, 
            L=2.*np.pi, 
            N=1024, 
            dt=0.001, 
            nu=0.02, 
            dforce=True,
            ssmforce=False,
            nsteps=None, 
            tend=5., 
            u0=None, 
            v0=None, 
            case=None, 
            forcing=False, 
            ssm=False, 
            dsm=False, 
            noise=0., 
            seed=42, 
            version=0, 
            nunoise=False, 
            numAgents=1,
            s=1):
        
        # Number of agents (>1 for MARL)
        self.numAgents = numAgents
        
        # SGS models
        assert( (ssm and dsm) == False )

        # Randomness
        np.random.seed(None)
        self.noise = noise*L
        self.offset = np.random.normal(loc=0., scale=self.noise) if self.noise > 0. else 0.
        while np.abs(self.offset) > L:
            self.offset = np.random.normal(loc=0., scale=self.noise) if self.noise > 0. else 0.

        self.s = s
        
        # seed for forcing
        np.random.seed(seed)
        
        # seed of turbulent IC 
        self.tseed = seed

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
        
        # Basis
        self.M = 0
        self.basis = None
        self.actions = None
        self.version = version

        # Static Smagorinsky Constant
        self.cs = 0.1
        self.ssm = ssm
        self.dsm = dsm

        # direct forcing or not
        self.dforce = dforce
        # static smagorinsky forcing
        self.ssmforce = ssmforce

        if self.ssmforce == True and self.dforce == False:
            print("[Burger] SSM forcing requires dforce")
            sys.exit()
 
        # time when field space transformed
        self.uut = -1
        # field in real space
        self.uu = None
        # interpolation of truth
        self.f_truth = None
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

    def setup_basis(self, M, kind = 'uniform'):
        self.M = M
        
        # Action record
        if M > 1:
            if kind == 'uniform':
                self.basis = np.zeros((self.M, self.N))
                for i in range(self.M):
                    assert self.N % self.M == 0, "[Burger] Something went wrong in basis setup"
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

    def IC(self, u0=None, v0=None, case='zero'):
        
        # Set initial condition
        if (v0 is None):
            if (u0 is None):
                    
                    # Gaussian initialization
                    if case == 'gaussian':
                        assert False, "Disabled"
                        sigma = self.L/8
                        u0 = gaussian(self.x+self.offset, mean=0.5*self.L, sigma=sigma)
                        
                    # Box initialization
                    elif case == 'box':
                        assert False, "Disabled"
                        u0 = np.abs(self.x+self.offset-self.L/2)<self.L/8
                    
                    # Sinus
                    elif case == 'sinus':
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
       
    def setGroundTruth(self, x, t, uu_truth):
        self.f_truth = interpolate.interp2d(x, t, uu_truth, kind='cubic')
 
    def mapGroundTruth(self):
        t = np.arange(0,self.uu.shape[0])*self.dt
        return self.f_truth(self.x,t)

    def getAnalyticalSolution(self, t):
        print("[Burger] TODO.. exit")
        sys.exit()
 
    def step( self, actions=None ):

        Fforcing = np.zeros(self.N, dtype=np.complex64)

        if self.ssm == True:
                
            delta  = 2*np.pi/self.N
            dx2 = self.dx**2

            um = np.roll(self.u, 1)
            up = np.roll(self.u, -1)
            
            dudx = (self.u - um)/self.dx
            d2udx2 = (up - 2*self.u + um)/dx2

            nuSSM = (self.cs*delta)**2*np.abs(dudx)
            sgs = nuSSM*d2udx2 
            
            Fforcing += fft( sgs )

        if self.dsm == True:

         
            delta  = 2*np.pi/self.N
            deltah = 4*np.pi/self.N
            
            hidx = np.abs(self.k)>self.N//4
            dx2 = self.dx**2

            v2 = fft(self.u**2)

            v2h = v2
            v2h[hidx] = 0
            L1 = 0.5*np.real(ifft(v2h))

            vh = self.v
            vh[hidx] = 0

            uh = np.real(ifft(vh))
            L2 = 0.5*uh**2
            L = L1-L2

            um = np.roll(self.u, 1)
            up = np.roll(self.u, -1)
            
            dudx = (self.u - um)/self.dx
            d2udx2 = (up - 2*self.u + um)/dx2
            
            w2 = fft(np.abs(dudx)*dudx)
            w2h = w2
            w2h[hidx] = 0
            M1 = delta**2*np.real(ifft(w2h))

            uhm = np.roll(uh, 1)
            duhdx = (uh - uhm)/self.dx
            M2 = deltah**2*np.abs(duhdx)*duhdx

            M = M1 - M2
            csd = L/M

            H = -L
            malt = 4./deltah**2*M2 - 1./delta**2*M1
            Malt = (malt-np.roll(malt,1))/self.dx
            csd2alt = np.mean(H*Malt)/np.mean(Malt*Malt)
            nuDSMalt = csd2alt*np.abs(dudx)
            sgsalt = nuDSMalt*d2udx2
            
            nuDSM = (csd*delta)**2*np.abs(dudx)
            sgs = nuDSM*d2udx2
            
            #Fforcing += fft( sgs )
            Fforcing += fft( sgsalt )

        if self.forcing:
        
            forcing = np.zeros(self.N)
         
            A=np.sqrt(2.)/self.L
            for k in range(1,4):
                ridx = self.ioutnum % self.s
                r1 = self.randfac1[k, ridx]
                r2 = self.randfac2[k, ridx] 
                forcing += r1*A/np.sqrt(k*self.s*self.dt)*np.cos(2*np.pi*k*(self.x+self.offset)/self.L+2*np.pi*r2);

            Fforcing = fft( forcing )
        
            self.f[self.ioutnum, :] = forcing
            """
            hidx = (np.abs(self.k)>70) 
            z = self.v.copy()
            z[hidx] = 0
            energy = sum(z**2)
            eta = 1/2048
            eps = self.nu**3/eta**4
            gamma = eps / energy
            Fforcing = -gamma * z
            """
            
        if (actions is not None):
 
            actions = actions if self.numAgents == 1 else [a for acs in actions for a in acs]
            
            assert self.basis is not None, "[Burger] Basis not set up (is None)."
            assert len(actions) == self.M, "[Burger] Wrong number of actions (provided {}/{}".format(len(actions), self.M)
            
            forcing = np.matmul(actions, self.basis)
            self.actionHistory[self.ioutnum,:] = forcing
            
            if self.dforce == False:
                u = self.uu[self.ioutnum,:]
                up = np.roll(u,1)
                um = np.roll(u,-1)
                d2udx2 = (up - 2.*u + um)/self.dx**2
                forcing *= d2udx2
            
            if self.ssmforce == True:
                delta  = 2*np.pi/self.N
                dx2 = self.dx**2

                um = np.roll(self.u, 1)
                up = np.roll(self.u, -1)
                
                dudx = (self.u - um)/self.dx
                d2udx2 = (up - 2*self.u + um)/dx2

                nuSSM = (forcing*delta)**2*np.abs(dudx)
                ssm = nuSSM*d2udx2 
            
            self.sgsHistory[self.ioutnum,:] = forcing
            Fforcing += fft( forcing )

        """
        RK3 in time
        """
        """
        v1 = self.v + self.dt * (-0.5*self.k1*fft(self.u**2) + self.nu*self.k2*self.v + Fforcing)
        u1 = np.real(ifft(v1))
        
        v2 = 3./4.*self.v + 1./4.*v1 + 1./4. * self.dt * (-0.5*self.k1*fft(u1**2) + self.nu*self.k2*v1 + Fforcing)
        u2 = np.real(ifft(v2))

        v3 = 1./3.*self.v + 2./3.*v2 + 2./3. * self.dt * (-0.5*self.k1*fft(u2**2) + self.nu*self.k2*v2 + Fforcing)
        self.v = v3
        """

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

    def getMseReward(self, shift):

        try:
            newx = self.x + shift
            newx[newx>self.L] = newx[newx>self.L] - self.L
            newx[newx<0] = newx[newx<0] + self.L
            midx = np.argmax(newx)
            if midx == len(newx)-1:
                uTruthToCoarse = self.f_truth(newx, self.t)
            else:
                uTruthToCoarse = np.concatenate(((self.f_truth(newx[:midx+1], self.t)), self.f_truth(newx[midx+1:], self.t)))
            uDiffMse = ((uTruthToCoarse - self.uu[self.ioutnum,:])**2)

        except FloatingPointError:
            print("[Burger] Floating point exception occured in mse", flush=True)
            return -np.inf*np.ones(self.numAgents)

        rewards = np.zeros(self.numAgents)
        for agentId in range(self.numAgents):
            a = agentId*self.N//self.numAgents
            b = (agentId+1)*self.N//self.numAgents
            rewards[agentId] = -uDiffMse[a:b].mean()
        
        return rewards
 
     
    def getState(self, nAgents = None):
        # Convert from spectral to physical space
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
                state = np.vstack((dudt,d2udx2))
            elif self.version == 2:
                state = np.vstack((u,u**2))
            elif self.version == 3:
                state = d2udx2
            else:
                print("[Burger] Version not recognized", flush=True)
                sys.exit()


        except FloatingPointError:

            print("[Burger] Floating point exception occured in getState", flush=True)
            if self.version == 0:
                state = np.inf*np.ones(self.N)
            elif self.version == 1:
                state = np.inf*np.ones((2,self.N))
            elif self.version == 2:
                state = np.inf*np.ones((2,self.N))
            elif self.version == 3:
                state = np.inf*np.ones(self.N)
            else:
                print("[Burger] Version not recognized", flush=True)
                sys.exit()
       
        states = []

        if self.numAgents == 1:
            states = [state.flatten().tolist()]
            if self.version == 3:
                ek = 1./2.*np.real( self.v.conj()*self.v / self.N ) * self.dx
                states[0] += ek[:self.N//2].tolist()

        else:
            for agentId in range(self.numAgents):
                a = agentId*self.N//self.numAgents - 1
                b = (agentId+1)*self.N//self.numAgents + 1
                index = np.arange(a,b) % self.N
                
                if self.version == 0:
                    states.append(state[index].flatten().tolist())
                elif self.version == 1:
                    states.append(state[:,index].flatten().tolist())
                elif self.version == 2:
                    states.append(state[:,index].flatten().tolist())
                elif self.version == 3:
                    ek = 1./2.*np.real( self.v.conj()*self.v / self.N ) * self.dx
                    states.append(state[index].tolist() + ek[:self.N//2].tolist())
                else:
                    print("[Burger] Version not recognized", flush=True)
                    sys.exit()
            
        return states

    def compute_Sgs(self, nURG):
        hidx = np.abs(self.k)>nURG//2
        self.sgsHistory = np.zeros(self.uu.shape)
        self.sgsHistoryAlt = np.zeros(self.uu.shape)
        self.sgsHistoryAlt2 = np.zeros((self.stepnum+1, nURG))
    
        r = nURG/self.N

        for idx in range(self.uu.shape[0]):
            dtidx = idx+1 if idx < self.uu.shape[0]-1 else idx-1

            # calc uhat(t+1)
            upt = self.uu[dtidx,:]
            vpt = fft(upt)
            vpth = vpt
            vpth[hidx] = 0 #filter
            uhpt = np.real(ifft(vpth))

            uhptAlt2 = np.real(ifft(np.concatenate((vpt[:(nURG+1)//2],vpt[-(nURG-1)//2:]))))*r

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
            u2h = np.real(ifft(v2h))
            
            uhAlt2 = np.real(ifft(np.concatenate((v[:(nURG+1)//2],v[-(nURG-1)//2:]))))*r

            duhdt = (uhpt-uh)/self.dt
            duhdtAlt2 = (uhptAlt2-uhAlt2)/self.dt
            if (idx == self.uu.shape[0]-1):
                duhdt *= -1
                duhdtAlt2 *= -1

            uhp = np.roll(uh,-1)
            uhm = np.roll(uh,+1)
            u2hm = np.roll(u2h,+1)

            uhpAlt2 = np.roll(uhAlt2,-1)
            uhmAlt2 = np.roll(uhAlt2,+1)

            # calc latteral derivatives
            duhdx = (uh - uhm)/self.dx
            d2uhdx2 = (uhp-2.*uh+uhm)/self.dx**2

            du2hdx = (u2h - u2hm)/self.dx
            
            duhdxAlt2 = (uhAlt2-uhmAlt2)/self.dx*r
            d2uhdx2Alt2 = (uhpAlt2-2.*uhAlt2+uhmAlt2)/self.dx**2*r**2

            self.sgsHistory[idx,:] = -uh*duhdx  + 0.5*du2hdx
            self.sgsHistoryAlt[idx,:] = duhdt + uh*duhdx - self.nu*d2uhdx2
            self.sgsHistoryAlt2[idx,:] = duhdtAlt2 + uhAlt2*duhdxAlt2 - self.nu*d2uhdx2Alt2
