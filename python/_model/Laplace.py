import sys
import time
from numpy import pi
from scipy import interpolate
from scipy.sparse import diags
import numpy as np

class Laplace:   
    #
    # Solution to the Laplace equation
    #
    # u_xx = f(x)

    def __init__(self, L=2.*np.pi, N=512, dt=0.001, ic='one', sforce='zero', episodeLength=500, noise=0., version=0):
        
        # Initialize
        self.N  = int(N+2)
        self.L  = float(L); 
        self.dt = float(dt); 
        self.nsteps = int(episodeLength)
        self.nout   = self.nsteps
        self.offset = np.random.normal(0., self.L*noise) if noise > 0. else 0.
        self.version = version 
        
        # save to self
        self.dx = L/(N+1)
        self.x  = np.linspace(0, self.L, self.N, endpoint=True)
        assert self.x[1]-self.x[0] == self.dx, print(self.x, self.dx)

        self.__setup_timeseries()

        self.ic = ic
        self.sforce = sforce
        self.IC(ic=ic, sforce=sforce)
        
    def __setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        
        # nout+1 because we store the IC as well
        self.uu = np.zeros([self.nout+1, self.N])
        self.tt = np.zeros(self.nout+1)
        self.gradientHistory = np.zeros([self.nout+1, self.N])
        self.actionHistory0 = np.zeros([self.nout+1, self.N])
        self.actionHistory1 = np.zeros([self.nout+1, self.N])
        self.actionHistory2 = np.zeros([self.nout+1, self.N])
        
    def IC(self, ic='one', sforce='zero'):
        
        if ic == 'zero':
            self.u0  = np.zeros(self.N)
        elif ic == 'one':
            self.u0  = np.ones(self.N)
        elif ic == 'sin':
            self.u0  = 1.+np.sin(self.x)
        elif ic == 'cos':
            self.u0  = np.cos(self.x)
        else:
            print("[Laplace] Error: ic unknown")
            sys.exit()

        # Box initialization
        if sforce == 'zero':
            force = np.zeros(self.N)
        
        # Sinus
        elif sforce == 'sin':
            force = np.sin((self.x - self.offset)*2*np.pi/self.L)

        # Sinus & Cosinus
        elif sforce == 'sincos':
            if np.random.rand() > 0.5:
                force = np.sin((self.x - self.offset)*2*np.pi/self.L)
            else:
                force = np.cos((self.x - self.offset)*2*np.pi/self.L)

        elif sforce == 'fourier':
            rand = np.random.rand()
            if rand > 0.75:
                force = np.sin((self.x - self.offset)*2*np.pi/self.L)
            elif rand > 0.5:
                force = np.cos((self.x - self.offset)*2*np.pi/self.L)
            elif rand > 0.25:
                force = np.sin((self.x - self.offset)*4*np.pi/self.L)
            else:
                force = np.cos((self.x - self.offset)*4*np.pi/self.L)
        
        # Gaussian
        elif sforce == 'gaussian':
            force = np.exp(-0.5*(0.5*self.L - self.x + self.offset)**2)

        else:
            print(f"[Laplace] Error: force {sforce} unknown")
            sys.exit()

        # and save to self
        self.u   = self.u0
        self.force = force
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0
  
        # store the IC in [0]
        self.uu[0,:] = self.u0
        self.tt[0]   = 0.
         
        # initial gradient
        um = np.roll(self.u, 1)
        up = np.roll(self.u, -1)
        grad = (-2.*self.u + um + up)/(self.dx**2)
        grad[0] = (-2.*self.u[0] + self.u[1] + self.u[-2])/(self.dx**2)
        grad[-1] = (-2.*self.u[-1] + self.u[1] + self.u[-2])/(self.dx**2)
 
        self.gradientHistory[0, :] = grad
      
    def step( self, actions, numAgents):
        
        assert(numAgents == self.N-2)
        M = np.zeros((self.N, self.N))
        M[0,0] = 1
        M[-1,-1] = 1
        for i in range(numAgents):
            assert len(actions[i]) == 3, f"[Laplace] action len not 3, it is {len(actions)}"
            M[i+1,i] = actions[i][0]
            M[i+1,i+1] = actions[i][1]
            M[i+1,i+2] = actions[i][2]

        d2udx2 = M @ self.u
        #self.u = self.u + self.dt * d2udx2 / self.dx**2
        self.u = self.u + self.dt * d2udx2

        self.u[0] = 1.
        self.u[-1] = 1.

        self.stepnum += 1
        self.t       += self.dt
 
        self.ioutnum += 1
        self.uu[self.ioutnum,:] = self.u
        self.tt[self.ioutnum]   = self.t

        um = np.roll(self.u, 1)
        up = np.roll(self.u, -1)
        grad = (-2.*self.u + um + up)/(self.dx**2)
        grad[0] = (-2.*self.u[0] + self.u[1] + self.u[-2])/(self.dx**2)
        grad[-1] = (-2.*self.u[-1] + self.u[1] + self.u[-2])/(self.dx**2)
 
        self.gradientHistory[self.ioutnum, :] = grad
        self.actionHistory0[self.ioutnum,:] = np.append(M[0,-1], np.diagonal(M, offset=-1))
        self.actionHistory1[self.ioutnum,:] = np.diagonal(M, offset=0) 
        self.actionHistory2[self.ioutnum,:] = np.append(M[-1,0] , np.diagonal(M, offset=1))

        
    def getDirectReward(self, numAgents=1):
        assert numAgents == self.N-2, f"[Laplace] direct reward neeeds N-2 agents (using {numAgents})"
        um = np.roll(self.u, 1)
        up = np.roll(self.u, -1)
        d2udx2 = (-2.*self.u + um + up)/(self.dx**2)
        return (-np.power(d2udx2-self.force,2))[1:-1].tolist()

    def getState(self, numAgents=1):
        assert(self.N-2 == numAgents)
        state = [ self.u[i:i+3].tolist() + [self.force[i+1]] for i in range(numAgents)]

        return state
