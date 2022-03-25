import sys
from numpy import pi
from scipy import interpolate
#from scipy.fftpack import fft, ifft, fftfreq
import numpy as np

import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, random

np.seterr(over='raise', invalid='raise')
def gaussian( x, mean, sigma ):
    return 1/jnp.sqrt(2*jnp.pi*sigma**2)*jnp.exp(-1/2*( (x-mean)/sigma )**2)

def hat( x, mean, dx ):
    left  = jnp.clip((x + dx - mean)/dx, a_min = 0., a_max = 1.)
    right = jnp.clip((dx - x + mean)/dx, a_min = 0., a_max = 1.)
    return left + right - 1.

class Burger_jax:
    #
    # Solution of the Burgers equation
    #
    # u_t + u*u_x = nu*u_xx0
    # with periodic BCs on x \in [0, L]: u(0,t) = u(L,t).

    def __init__(self, L=1./(2.*jnp.pi), N=128, dt=0.25, nu=0.0, nsteps=None, tend=150, u0=None, v0=None, case=None, noisy = False):

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
        self.dx     = L/N
        self.x      = jnp.linspace(0, self.L, N, endpoint=False)
        self.dt     = dt
        self.nu     = nu
        self.nsteps = nsteps
        self.nout   = nsteps

        # Basis
        self.M = 0
        self.basis = None

        # gradient
        self.gradient = 0

        # time when field space transformed
        self.uut = -1
        # field in real space
        self.uu = None
        # ground truth in real space
        self.uu_truth = None
        # interpolation of truth
        self.f_truth = None

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

        # initialize simulation arrays
        self.__setup_timeseries()

        # precompute Fourier-related quantities
        self.__setup_fourier()



    def __setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)

        # nout+1 because we store the IC as well
        self.uu = jnp.zeros([self.nout+1, self.N])
        self.vv = jnp.zeros([self.nout+1, self.N], dtype=jnp.complex64)
        self.tt = jnp.zeros(self.nout+1)

        # store the IC in [0]
        self.uu = self.uu.at[0,:].set(self.u0)
        self.vv = self.vv.at[0,:].set(self.v0)
        self.tt = self.tt.at[0].set(0.)

    def __setup_fourier(self):
        self.k   = jnp.fft.fftfreq(self.N, self.L / (2*jnp.pi*self.N))
        self.k1  = 1j * self.k
        self.k2  = self.k1**2

    def setup_basis(self, M, kind = 'uniform'):
        self.M = M
        if M > 1:
            if kind == 'uniform':
                self.basis = jnp.zeros((self.M, self.N))
                for i in range(self.M):
                    assert self.N % self.M == 0, "[Burger] Something went wrong in basis setup"
                    idx1 = i * self.N//self.M
                    idx2 = (i+1) * self.N//self.M
                    self.basis = self.basis.at[i,idx1:idx2].set(1.)
            elif kind == 'hat':
                self.basis = jnp.ones((self.M, self.N))
                dx = self.L/(self.M-1)
                for i in range(self.M):
                    mean = i*dx
                    self.basis = self.basis.at[i,:].set(hat( self.x, mean, dx ))

            else:
                print("[Burger] Basis function not known, exit..")
                sys.exit()
        else:
            self.basis = jnp.ones((self.M, self.N))

        jnp.allclose(jnp.sum(self.basis, axis=0), 1)
        "setting up gradient"
        assert self.N % self.M == 0, "[Burger] Something went wrong in gradient setup"
        self.gradient = np.ones((self.N, self.M))

    def IC(self, u0=None, v0=None, case='box', seed=42):

        # Set initial condition
        if (v0 is None):
            if (u0 is None):

                    key = random.PRNGKey(42)
                    offset = self.dx*random.normal(key, shape=(16,)) if self.noisy else 0.

                    # Gaussian initialization
                    if case == 'gaussian':
                        # Gaussian noise (according to https://arxiv.org/pdf/1906.07672.pdf)
                        #u0 = jnp.random.normal(0., 1, self.N)
                        sigma = self.L/8
                        u0 = gaussian(self.x, mean=0.5*self.L+offset, sigma=sigma)

                    # Box initialization
                    elif case == 'box':
                        u0 = jnp.abs(self.x-self.L/2-offset)<self.L/8

                    # Sinus
                    elif case == 'sinus':
                        u0 = jnp.sin(self.x+offset)

                    else:
                        print("[Burger] Error: IC case unknown")
                        return -1

            else:
                # check the input size
                if (jnp.size(u0,0) != self.N):
                    print("[Burger] Error: wrong IC array size")
                    return -1
                else:
                    # if ok cast to jnp.array
                    u0 = jnp.array(u0)
            # in any case, set v0:
            v0 = jnp.fft.fft(u0)
        else:
            # the initial condition is provided in v0
            # check the input size
            if (jnp.size(v0,0) != self.N):
                print("[Burger] Error: wrong IC array size")
                return -1
            else:
                # if ok cast to jnp.array
                v0 = jnp.array(v0)
                # and transform to physical space
                u0 = jnp.real(jnp.fft.ifft(v0))

        # and save to self
        self.u0  = u0
        self.u   = u0
        self.v0  = v0
        self.v   = v0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition

    def setGroundTruth(self, t, x, uu):
        self.uu_truth = uu
        self.f_truth = interpolate.interp2d(x, t, self.uu_truth, kind='cubic')

    def mapGroundTruth(self):
        t = jnp.arange(0,self.uu.shape[0])*self.dt
        return self.f_truth(self.x,t)

    def getAnalyticalSolution(self, t):
        print("[Diffusion] TODO.. exit")
        sys.exit()

    def expl_euler(self, actions, u, v):

        Fforcing = jnp.zeros(self.N)
        if (actions is not None):
            assert self.basis is not None, "[Burger] Basis not set up (is None)."
            assert len(actions) == self.M, "[Burger] Wrong number of actions (provided {}/{}"
            forcing = jnp.matmul(actions, self.basis)
            Fforcing = jnp.fft.fft( forcing )

        # explicit euler
        #v = v - self.dt*0.5*self.k1*jnp.fft.fft(u**2) + self.dt*self.nu*self.k2*v + self.dt*Fforcing
        v = v - self.dt*0.5*self.k1*jnp.fft.fft(u*u) + self.dt*self.nu*self.k2*v + self.dt*Fforcing
        u = jnp.real(jnp.fft.ifft(v))

        return (u, v)


    def grad(self, actions, u, v):

        #ee = jit(self.expl_euler)
        if (actions is not None):
            return jacfwd(self.expl_euler, has_aux=True)(actions, u, v)

    def step(self, actions):

        #ee = jit(self.expl_euler)
        (self.u, self.v) = self.expl_euler(actions, self.u, self.v)
        self.gradient = self.grad(actions, self.u, self.v)[0]

        self.stepnum += 1
        self.t       += self.dt

        self.ioutnum += 1

        self.uu = self.uu.at[self.ioutnum,:].set(self.u)
        self.vv = self.vv.at[self.ioutnum,:].set(self.v)
        self.tt = self.tt.at[self.ioutnum].set(self.t)


    def simulate(self, nsteps=None, restart=False, correction=[]):
        #
        # If not provided explicitly, get internal values
        actions = jnp.zeros(self.M)

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
                    self.step(actions)
            else:
                for n in range(1,self.nsteps+1):
                    self.step(actions)
                    self.v += correction

        except FloatingPointError:
            print("[Burger] Floating point exception occured")
            flush = True
            # something exploded
            # cut time series to last saved solution and return
            self.nout = self.ioutnum
            self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
            self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
            return -1

    def fou2real(self):
        # Convert from spectral to physical space
        #self.uut = self.stepnum
        self.uu = jnp.real(jnp.fft.ifft(self.vv))

    def compute_Ek(self):
        #
        # compute all forms of kinetic energy
        #
        # Kinetic energy as a function of wavenumber and time
        self.__compute_Ek_kt()

        # Time-averaged energy spectrum as a function of wavenumber
        self.Ek_k = jnp.sum(self.Ek_kt, 0)/(self.ioutnum+1) # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0]

        # Total kinetic energy as a function of time
        self.Ek_t = jnp.sum(self.Ek_kt, 1)

        # Time-cumulative average as a function of wavenumber and time
        self.Ek_ktt = jnp.cumsum(self.Ek_kt, 0)[:self.ioutnum+1,:] / jnp.arange(1,self.ioutnum+2)[:,None] # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero

        # Time-cumulative average as a function of time
        self.Ek_tt = jnp.cumsum(self.Ek_t, 0)[:self.ioutnum+1] / jnp.arange(1,self.ioutnum+2) # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero

    def __compute_Ek_kt(self):
        try:
            self.Ek_kt = 1./2.*jnp.real( self.vv.conj()*self.vv / self.N ) * self.dx
        except FloatingPointError:
            #
            # probable overflow because the simulation exploded, try removing the last solution
            problem=True
            remove=1
            self.Ek_kt = jnp.zeros([self.nout+1, self.N]) + 1e-313
            while problem:
                try:
                    self.Ek_kt[0:self.nout+1-remove,:] = 1./2.*jnp.real( self.vv[0:self.nout+1-remove].conj()*self.vv[0:self.nout+1-remove] / self.N ) * self.dx
                    problem=False
                except FloatingPointError:
                    remove+=1
                    problem=True
        return self.Ek_kt

    def space_filter(self, k_cut=2):
        #
        # spatially filter the time series
        self.uu_filt  = jnp.zeros([self.nout+1, self.N])
        for n in range(self.nout+1):
            v_filt = jnp.copy(self.vv[n,:])    # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[jnp.abs(self.k)>=k_cut] = 0 # set to zero wavenumbers > k_cut
            self.uu_filt[n,:] = jnp.real(jnp.fft.ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt

    def space_filter_int(self, k_cut=2, N_int=10):
        #
        # spatially filter the time series
        self.N_int        = N_int
        self.uu_filt      = jnp.zeros([self.nout+1, self.N])
        self.uu_filt_int  = jnp.zeros([self.nout+1, self.N_int])
        self.x_int        = 2*pi*self.L*jnp.r_[0:self.N_int]/self.N_int
        for n in range(self.nout+1):
            v_filt = jnp.copy(self.vv[n,:])   # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[jnp.abs(self.k)>=k_cut] = 313e6
            v_filt_int = v_filt[v_filt != 313e6] * self.N_int/self.N
            self.uu_filt_int[n,:] = jnp.real(jnp.fft.ifft(v_filt_int))
            v_filt[jnp.abs(self.k)>=k_cut] = 0
            self.uu_filt[n,:] = jnp.real(jnp.fft.ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt

    def getReward(self):
        # Convert from spectral to physical space
        t = [self.t]
        uMap = self.f_truth(self.x, t)
        return -jnp.abs(self.u-uMap)

    def getState(self, nAgents = None):
        # Convert from spectral to physical space
        self.fou2real()

        # Extract state
        u = self.uu[self.ioutnum,:]
        #dudu = jnp.zeros(self.N)
        #dudu[:-1] = (u[1:]-u[:-1])/self.dx
        #dudu[-1] = dudu[-2]
        dudt = (self.uu[self.ioutnum,:]-self.uu[self.ioutnum-1,:])/self.dt
        #state = jnp.column_stack( (u, dudu, dudt) )
        state = jnp.column_stack( (u, dudt) )
        return state

    def updateField(self, factors):
        # elementwise multiplication
        self.v = self.v * factors
