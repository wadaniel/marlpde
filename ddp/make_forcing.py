# This is to compute dataset for analysis
import os
import pickle

# Calculates filtered field 'u_bar' and filtered forcing term 'f_bar', and SG term PI
def calc_bar(U_DNS, f_store, NX, NY):
    Lx=100

    f_bar = filter_bar(f_store,NX,NY)
    u_bar = filter_bar(U_DNS,NX,NY)

    U2_DNS = U_DNS**2
    u2_bar = filter_bar(U2_DNS,NX,NY)
    
    tau = .5*(u2_bar - u_bar**2)
    mtau = np.roll(tau, 1)
 
    dx = Lx/NY
    PI = (tau-mtau)/dx

    return (u_bar, PI, f_bar)

scratch = os.getenv("SCRATCH", default=".")

print("Loading data..")
U_DNS = pickle.load( open('{}/DNS_Burgers_s_20.pickle'.format(scratch), 'rb') )
f_store = pickle.load( open('{}/DNS_Force_LES_s_20.pickle'.format(scratch), 'rb') )

u_bar,PI,f_bar = calc_bar(U_DNS[:,1::20],f_store,1024,128)

pickle.dump(u_bar, open('{}/u_bar_all_regions.pickle'.format(scratch), 'wb'))
pickle.dump(PI, open('{}/PI_all_regions.pickle'.format(scratch), 'wb'))
pickle.dump(PI, open('{}/f_bar_all_regions.pickle'.format(scratch), 'wb'))



