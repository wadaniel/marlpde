import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('./../../_model/')

from burger_analytical import *

#burgers_viscous_time_exact1_test01()



nu = 0.01

L = 2*np.pi
vxn = 256
vx = np.linspace(0.,L,vxn)

tEnd = 5
vtn = 100
vt = np.linspace(0.,tEnd, vtn)
dt = tEnd/vtn


f0 = lambda x: np.array(np.abs(x-L/2)<L/8, dtype=float)
result = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt, f0 )

print("Plotting advection_evolution.png ...")
fig, axs = plt.subplots(4,4, sharex=True, sharey=False, figsize=(15,15))
for i in range(16):
    t = i * tEnd / 16
    tidx = int(t/dt)
    k = int(i / 4)
    l = i % 4
    axs[k,l].plot(vx, result[:,tidx], '--k')

fig.savefig('burger_evolution.png'.format())
plt.close()
