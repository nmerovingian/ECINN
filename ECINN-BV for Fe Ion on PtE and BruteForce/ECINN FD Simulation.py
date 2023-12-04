import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import cm
from helper import flux_sampling,expParameters
from sklearn.metrics import mean_squared_error,mean_absolute_error

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



colors = cm.viridis(np.linspace(0,1,len(scan_rates)))


fig,ax = plt.subplots(figsize=(8,4.5))
for index in range(len(scan_rates)):
    sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_files[0])
    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
    time_array = np.linspace(0,maxT,num=num_test_samples)


    df_exp = pd.read_csv(exp_dimensionless_files[index])
    df_ECINN_FD  = pd.read_csv(ECINN_FD_dimensionless_files[index])

    df_exp_sampled = flux_sampling(time_array,df_exp,maxT)
    df_ECINN_FD_sampled = flux_sampling(time_array,df_ECINN_FD,maxT)



    exp_forward_peak_J = df_exp.iloc[:,1].min()
    exp_reverse_peak_J = df_exp.iloc[:,1].max()
    exp_forward_peak_theta =df_exp.iloc[df_exp.iloc[:,1].idxmin(),0]
    exp_reverse_peak_theta =df_exp.iloc[df_exp.iloc[:,1].idxmax(),0]

    ECINN_FD_forward_peak_J = df_ECINN_FD.iloc[:,1].min()
    ECINN_FD_reverse_peak_J = df_ECINN_FD.iloc[:,1].max()

    MSE = mean_squared_error(df_exp_sampled,df_ECINN_FD_sampled)

    MAE = mean_absolute_error(df_exp_sampled,df_ECINN_FD_sampled)
    epsilon_J = MAE/exp_forward_peak_J
    epsilon_J_max = np.max(np.abs(df_exp_sampled-df_ECINN_FD_sampled))/exp_forward_peak_J
    epsilon_J_peak_forward = np.abs(ECINN_FD_forward_peak_J-exp_forward_peak_J)/exp_forward_peak_J
    epsilon_J_peak_reverse = np.abs(ECINN_FD_reverse_peak_J-exp_reverse_peak_J)/exp_reverse_peak_J
    epsilon_J_peak = np.abs(epsilon_J_peak_forward)*0.5 + np.abs(epsilon_J_peak_reverse)*0.5



    ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=2,label=f'$\\nu=${scan_rates[index]:.2f} V/s\n$MSE={MSE:.3f}$, $\epsilon_{{J_{{peak}}}}$={epsilon_J_peak:.2%}',color=tuple(colors[index]))
    #ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=2,label=f'$\\nu=${scan_rates[index]:.2f} V/s, $MSE={MSE:.3f}$, MAE={MAE:.3f}\n$\epsilon_{{J}}$={epsilon_J:.2%}, $\epsilon_{{J_{{max}}}}$={epsilon_J_max:.2%}\n$\epsilon_{{J_{{peak,forward}}}}$={epsilon_J_peak_forward:.2%} $\epsilon_{{J_{{peak,reverse}}}}$={epsilon_J_peak_reverse:.2%}\n$\epsilon_{{J_{{peak}}}}$={epsilon_J_peak:.2%} $\epsilon_{{\\theta_{{peak}}}}$={epsilon_theta_peak:.2%}\n$\epsilon_{{\\theta_{{peak,forward}}}}$={epsilon_theta_peak_forward:.2%} $\epsilon_{{\\theta_{{peak,reverse}}}}$={epsilon_theta_peak_reverse:.2%}',color=tuple(colors[index]))
    ax.plot(df_ECINN_FD.iloc[:,0],df_ECINN_FD.iloc[:,1],ls='--',lw=2,color=tuple(colors[index]))


ax.legend()
ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')
fig.savefig("ECINN-FD.png",dpi=250,bbox_inches='tight')





    
