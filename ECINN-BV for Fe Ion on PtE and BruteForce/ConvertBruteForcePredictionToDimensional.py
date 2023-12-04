import pandas as pd
import numpy as np 
from ConvertToDimensionless import ERef, r_e,Dref
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from helper import flux_sampling,expParameters
from sklearn.metrics import mean_squared_error,mean_absolute_error
from FD_Simulation_Unequal_D.FD_Simulation import simulation

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
num_test_samples = 1000

BruteForceResults = pd.read_csv("BruteForceResults.csv")
K0 = BruteForceResults['K0'].iloc[0]
alpha = BruteForceResults['alpha'].iloc[0]
beta = BruteForceResults['beta'].iloc[0]
davg = BruteForceResults['davg'].iloc[0]

k0 = K0*Dref/r_e
k0_eff_c = k0*math.exp(alpha*ERef*96485/8.314/298)
k0_eff_a = k0*math.exp(-beta*ERef*96485/8.314/298)
Davg = Dref * davg

print(f'k0={k0:.3E}, kc={k0_eff_c:.3E}, ka={k0_eff_a:.3E}, davg={Davg}' )




colors = cm.viridis(np.linspace(0,1,len(scan_rates)))

fig,ax = plt.subplots(figsize=(8,4.5))
for index in range(len(scan_rates)):
    sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_files[index])
    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
    time_array = np.linspace(0,maxT,num=num_test_samples)

    FD_file_name = simulation(sigma=sigma,theta_i=theta_i,theta_v=theta_v,K0=K0,alpha=alpha,beta=beta,dA=davg,dB=davg)


    df_exp = pd.read_csv(exp_dimensionless_files[index])
    df_BruteForce_FD  = pd.read_csv(FD_file_name)

    df_exp_sampled = flux_sampling(time_array,df_exp,maxT)
    df_BruteForce_FD_sampled = flux_sampling(time_array,df_BruteForce_FD,maxT)



    exp_forward_peak_J = df_exp.iloc[:,1].min()
    exp_reverse_peak_J = df_exp.iloc[:,1].max()


    ECINN_FD_forward_peak_J = df_BruteForce_FD.iloc[:,1].min()
    ECINN_FD_reverse_peak_J = df_BruteForce_FD.iloc[:,1].max()


    MSE = mean_squared_error(df_exp_sampled,df_BruteForce_FD_sampled)

    MAE = mean_absolute_error(df_exp_sampled,df_BruteForce_FD_sampled)
    epsilon_J = MAE/exp_forward_peak_J
    epsilon_J_max = np.max(np.abs(df_exp_sampled-df_BruteForce_FD_sampled))/exp_forward_peak_J
    epsilon_J_peak_forward = np.abs(ECINN_FD_forward_peak_J-exp_forward_peak_J)/exp_forward_peak_J
    epsilon_J_peak_reverse = np.abs(ECINN_FD_reverse_peak_J-exp_reverse_peak_J)/exp_reverse_peak_J
    epsilon_J_peak = np.abs(epsilon_J_peak_forward)*0.5 + np.abs(epsilon_J_peak_reverse)*0.5




    ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=2,label=f'$\\nu=${scan_rates[index]:.2f} V/s\n$MSE={MSE:.3f}$, $\epsilon_{{J_{{peak}}}}$={epsilon_J_peak:.2%}',color=tuple(colors[index]))

    ax.plot(df_BruteForce_FD.iloc[:,0],df_BruteForce_FD.iloc[:,1],ls='--',lw=2,color=tuple(colors[index]))


ax.legend()
ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')
fig.savefig("BruteForce-FD.png",dpi=250,bbox_inches='tight')



