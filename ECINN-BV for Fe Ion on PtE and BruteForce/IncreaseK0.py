import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import cm
from helper import expParameters
from FD_Simulation_Unequal_D.FD_Simulation import simulation


linewidth = 4
fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs





K0s  = [1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7]
colors = cm.viridis(np.linspace(0,1,len(K0s)))
alpha = 0.5
beta = 0.5
davg = 1.0

fig,ax = plt.subplots(figsize=(8,4.5))
theta_i = 20
theta_v = -20
sigma = 400

for index,K0 in enumerate(K0s):
        FD_file_name = simulation(sigma=sigma,theta_i=theta_i,theta_v=theta_v,K0=K0,alpha=alpha,beta=beta,dA=davg,dB=davg)
        df_FD = pd.read_csv(FD_file_name)
        ax.plot(df_FD.iloc[:,0],df_FD.iloc[:,1],lw=3,color=tuple(colors[index]),label=f'$K_0={K0:.0E}$',alpha=0.7)


ax.legend(ncols=2,fontsize=10)
ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')

fig.savefig('IncreaseK0.png',dpi=250,bbox_inches='tight')

