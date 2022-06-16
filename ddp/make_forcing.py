# This is to compute dataset for analysis
import os
import pickle
from helpers import *

scratch = os.getenv("SCRATCH", default=".")

print("Loading data..")
U_DNS = pickle.load( open('{}/DNS_Burgers_s_20.pickle'.format(scratch), 'rb') )
f_store = pickle.load( open('{}/DNS_Force_LES_s_20.pickle'.format(scratch), 'rb') )

u_bar,PI,f_bar = calc_bar(U_DNS[:,1::20],f_store,1024,128)

pickle.dump(u_bar, open('{}/u_bar_all_regions.pickle'.format(scratch), 'wb'))
pickle.dump(PI, open('{}/PI_all_regions.pickle'.format(scratch), 'wb'))
pickle.dump(PI, open('{}/f_bar_all_regions.pickle'.format(scratch), 'wb'))



