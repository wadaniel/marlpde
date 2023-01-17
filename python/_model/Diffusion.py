import sys
import time
from numpy import pi
from scipy import interpolate
from scipy.sparse import diags
import numpy as np

class Diffusion:  
    #
    # Solution to the Diffusion equation
    #
    # u_t = nu*u_xx
    # with periodic BCs on x \in [0, L]: u(0,t) = u(L,t).

    def __init__(self, L=2.*np.pi, N=512, dt=0.001, nu=0.01, nsteps=None, tend=5., case='box', version=0, noise=0., nunoise=False, seed=1337, implicit=False):
        
        # Randomness
        np.random.seed(None)
        self.seed = seed

        # EXplicit or Implicit euler schme
        self.implicit = implicit

        # Initialize
        self.L  = float(L); 
        self.dt = float(dt); 
        self.tend = float(tend)
        
        if (nsteps is None):
            nsteps = int(tend/dt)
        else:
            nsteps = int(nsteps)
            # override tend
            self.tend = dt*nsteps
        
        # save to self
        self.N  = N
        self.dx = L/N
        self.x  = np.linspace(0, self.L, N, endpoint=False)
        self.nu = nu

        if nunoise:
            self.nu = 0.01+0.02*np.random.uniform()

        self.noise = noise
        
        # Set initial condition
        self.offset = np.random.normal(loc=0., scale=self.noise) if self.noise > 0. else 0.
        
        self.nsteps = nsteps
        self.nout   = nsteps
 
        if (implicit == False and 2.*self.nu*self.dt >= self.dx**2):
            print("[Diffusion] Warning: CFL condition violated", flush=True)

        # Basis
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
        self.case = case

        if (case is not None):
            self.IC(case=case)
        else:
            print("[Diffusion] IC ambigous")
            sys.exit()
        
    def __setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        
        # nout+1 because we store the IC as well
        self.uu = np.zeros([self.nout+1, self.N])
        self.tt = np.zeros(self.nout+1)
        self.gradientHistory = np.zeros([self.nout+1, self.N])
        self.actionHistory = np.zeros([self.nout+1, self.N])
        self.solution = np.zeros([self.nout+1, self.N])
        
    def IC(self, case='box'):
        
           
        # Box initialization
        if case == 'box':
            u0 = np.zeros(self.N)
            u0[np.abs(self.x-self.L/2-self.offset)<self.L/8] = 1.
        
        # Sinus
        elif case == 'sinus':
            u0 = np.sin((self.x - self.offset)*2*np.pi/self.L)
        
        # Gaussian
        elif case == 'gaussian':
            u0 = np.exp(-0.5*(0.5*self.L + self.offset - self.x)**2)

        else:
            print("[Diffusion] Error: IC case unknown")
            sys.exit()

        # and save to self
        self.u0  = u0
        self.u   = u0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0
  
        # store the IC in [0]
        self.uu[0,:] = u0
        self.tt[0]   = 0.
        self.solution[0,:] = u0
       
    def setGroundTruth(self, t, x, uu):
        self.uu_truth = uu
        self.f_truth = interpolate.interp2d(x, t, self.uu_truth, kind='linear')
 
    def mapGroundTruth(self):
        return self.f_truth(self.x,self.tt)

    def FDstep(self):
        if self.implicit == True:
            """
            Impl. Euler Central Differences
            """
            c = self.dt*self.nu/(self.dx**2)
            M = diags([-c, 1+2*c, -c], [-1, 0, 1], shape=(self.N, self.N)).toarray()
            
            # periodic BC
            M[0,-1] = -c
            M[-1,0] = -c

            u = np.linalg.solve(M, self.u)
 

        else:
           """
           Expl. Euler Central Differences
           """
           um = np.roll(self.u, 1)
           up = np.roll(self.u, -1)
           d2udx2 = (-2.*self.u + um + up)/(self.dx**2)

           u = self.u + self.dt * self.nu * d2udx2

        return u
 
    def step( self, actions=None, numAgents=1):
        
        if (actions is None):
            self.u = self.FDstep()

        else:
            if numAgents == 1:
                assert len(actions) == 1, f"[Diffusion] action len not 1, it is {len(actions)}"
                ac = np.zeros(3)
                ac[0] = -actions[0]/2
                ac[1] = actions[0]
                ac[2] = -actions[0]/2
                M = diags(ac, [-1, 0, 1], shape=(self.N, self.N)).toarray()
                M[0,-1] = ac[0]
                M[-1,0] = ac[2]

            else:
                assert len(actions) == numAgents, f"[Diffusion] actions not of len numAgents"
                assert numAgents == self.N, f"[Diffusion] only works with {self.N} agents"
                M = np.zeros((self.N, self.N))
                for i in range(numAgents):
                    assert len(actions[i]) == 1, f"[Diffusion] action len not 1, it is {len(actions)}"
                    M[i,i] = actions[i][0]
                    
                    if i == 0:
                        M[i,-1] = -actions[i][0]/2
                        M[i,i+1] = -actions[i][0]/2
                    elif i == self.N-1:
                        M[i,i-1] = -actions[i][0]/2
                        M[i,0] = -actions[i][0]/2
                    else:
                        M[i,i+1] = -actions[i][0]/2
                        M[i,i-1] = -actions[i][0]/2

            d2udx2 = M @ self.u

            self.actionHistory[self.ioutnum,:] = np.diag(M)
            self.gradientHistory[self.ioutnum,:] = d2udx2
            self.u = self.u + self.dt * self.nu * d2udx2 / self.dx**2
        
        self.stepnum += 1
        self.t       += self.dt
 
        self.ioutnum += 1
        self.uu[self.ioutnum,:] = self.u
        self.tt[self.ioutnum]   = self.t
        
        if self.case == "sinus":
            self.solution[self.ioutnum, :] = self.u0*np.exp(-(2.*np.pi/self.L)**2*self.nu*self.t)


    def simulate( self, nsteps=None ):

        if nsteps is None:
            nsteps = self.nsteps

        # advance in time for nsteps steps
        try:
            for n in range(self.stepnum,self.nsteps):
                self.step()
                
        except FloatingPointError:
            print("[Burger] Floating point exception occured in simulate", flush=True)
            # something exploded
            # cut time series to last saved solution and return
            self.nout = self.ioutnum
            self.tt.resize(self.nout+1)           # nout+1 because the IC is in [0]
            self.uu.resize((self.N, self.nout+1)) # nout+1 because the IC is in [0]
            return -1

    def getMseReward(self, numAgents=1, offset=0.):
        assert(self.N % numAgents == 0)
         
        if self.case == "sinus":
            # analytical
            sol = self.getAnalyticalSolution(self.t)

            if (numAgents == 1):
                uDiffMse = ((sol - self.uu[self.ioutnum,:])**2).mean()
                return -uDiffMse

            else:
                section = int(self.N / numAgents)
                locDiffMse = [ -((sol[(i*section) : (i+1)*section] - self.uu[self.ioutnum,(i*section) : (i+1)*section])**2).mean() for i in range(numAgents) ]
                return locDiffMse
  
        else:
            # interpolation
            newx = self.x - offset
            newx[newx>self.L] = newx[newx>self.L] - self.L
            newx[newx<0] = newx[newx<0] + self.L
            midx = np.argmax(newx)

            if midx == len(newx)-1:
                uTruthToCoarse = self.f_truth(newx, self.t)
            else:
                uTruthToCoarse = np.concatenate(((self.f_truth(newx[:midx+1], self.t)), self.f_truth(newx[midx+1:], self.t)))

            if numAgents == 1:
                uDiffMse = ((uTruthToCoarse - self.uu[self.ioutnum,:])**2).mean()
                return -uDiffMse

            else:
                section = int(self.N / numAgents)
                locDiffMse = [ -((uTruthToCoarse[(i*section) : (i+1)*section] - self.uu[self.ioutnum,(i*section) : (i+1)*section])**2).mean() for i in range(numAgents) ]
                return locDiffMse
         
    def getState(self, numAgents=1):
        assert(self.N % numAgents == 0)

        if numAgents == 1:
            state = self.uu[self.ioutnum,:].tolist()
        else:
            section = int(self.N / numAgents)
            uextended = np.zeros(self.N+2)
            uextended[1:self.N+1] = self.uu[self.ioutnum, :]
            uextended[0] = self.uu[self.ioutnum, -1]
            uextended[-1] = self.uu[self.ioutnum, 0]

            state = [ uextended[(i*section) : 2+(i+1)*section].tolist() for i in range(numAgents)]

        return state


    def getAnalyticalSolution(self, t):
        if self.case == "sinus":
            return self.u0*np.exp(-(2.*np.pi/self.L)**2*self.nu*t)

        else:
            print(f"[Diffusion] case {self.case} not available")

