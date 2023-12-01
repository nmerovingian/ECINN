import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import cm
from helper import flux_sampling,expParameters
from sklearn.metrics import mean_squared_error

linewidth = 4
fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

scan_rates = [0.01,0.02,0.05]
exp_dimensionless_files = [
    "ExpData/Exp Dimensionless sigma=2.8137E+02 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=5.6273E+02 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=1.4068E+03 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",]
ECINN_FD_dimensionless_files = [
    "DataFD\sigma=2.81E+02 K0=3.04E+01 alpha=4.40E-01 beta=4.60E-01 kinetics=BV mode=linear,dA=5.26E-01,dB=5.26E-01 theta_i=1.43E+01 theta_v=-1.30E+01.csv",
    "DataFD\sigma=5.63E+02 K0=3.04E+01 alpha=4.40E-01 beta=4.60E-01 kinetics=BV mode=linear,dA=5.26E-01,dB=5.26E-01 theta_i=1.43E+01 theta_v=-1.30E+01.csv",
    "DataFD\sigma=1.41E+03 K0=3.04E+01 alpha=4.40E-01 beta=4.60E-01 kinetics=BV mode=linear,dA=5.26E-01,dB=5.26E-01 theta_i=1.43E+01 theta_v=-1.30E+01.csv",
]

num_test_samples = 1000

sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_files[0])
maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
time_array = np.linspace(0,maxT,num=num_test_samples)


colors = cm.viridis(np.linspace(0,1,len(scan_rates)))


fig,ax = plt.subplots(figsize=(8,4.5))
for index in range(len(scan_rates)):
    df_exp = pd.read_csv(exp_dimensionless_files[index])
    df_ECINN_FD  = pd.read_csv(ECINN_FD_dimensionless_files[index])

    df_exp_sampled = flux_sampling(time_array,df_exp,maxT)
    df_ECINN_FD_sampled = flux_sampling(time_array,df_ECINN_FD,maxT)
    MSE =mean_squared_error(df_exp_sampled,df_ECINN_FD_sampled)
    ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=2,label=f'$\\nu=${scan_rates[index]:.2f} V/s, MSE={MSE:.3f}',color=tuple(colors[index]))
    ax.plot(df_ECINN_FD.iloc[:,0],df_ECINN_FD.iloc[:,1],ls='--',lw=2,color=tuple(colors[index]))


ax.legend()
ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')
fig.savefig("ECINN-FD.png",dpi=250,bbox_inches='tight')


    
