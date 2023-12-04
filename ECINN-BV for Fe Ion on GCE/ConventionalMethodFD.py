import numpy as np
import matplotlib.pyplot as plt
from FD_Simulation_Unequal_D.FD_Simulation import simulation
from matplotlib import cm
from helper import exp_flux_sampling,toDimensionalPotential,toDimensionlessPotential,expParameters
import pandas as pd
from sklearn.metrics import mean_squared_error

linewidth = 4
fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs



scan_rates = [0.01,0.02,0.05] # V/s 
exp_dimensionless_files = [
    "ExpData/Exp Dimensionless sigma=8.7623E+02 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=1.7525E+03 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=4.3811E+03 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
]
num_test_samples = 1000
PortionAnalyzed = 0.75


sigmas =[]
Conventional_Method_FD_files = []
ECINN_FD_files = []
colors = cm.viridis(np.linspace(0,1,len(scan_rates)))

for index, exp_dimensionless_file in enumerate(exp_dimensionless_files):
    sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_file)
    sigmas.append(sigma)
    Cpnventional_FD_file_name = simulation(sigma=sigma,K0=0.254,alpha=0.38,beta=0.5,kinetics='BV',mode='linear',dA=0.389,dB=0.389,theta_i=theta_i,theta_v=theta_v,saving_directory='./DataFD')
    Conventional_Method_FD_files.append(Cpnventional_FD_file_name)
    ECINN_FD_file_name = simulation(sigma=sigma,K0=0.254,alpha=0.339,beta=0.5,kinetics='BV',mode='linear',dA=0.590,dB=0.590,theta_i=theta_i,theta_v=theta_v,saving_directory='./DataFD')
    ECINN_FD_files.append(ECINN_FD_file_name)

fig,axs= plt.subplots(figsize=(8,9),nrows=2)
for index in range(len(scan_rates)):

    sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_files[index])
    FullScanT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
    maxT = FullScanT*PortionAnalyzed
    time_array = np.linspace(0,maxT,num=num_test_samples)    
    
    df_exp = pd.read_csv(exp_dimensionless_files[index])
    df_Conventional_FD  = pd.read_csv(Conventional_Method_FD_files [index])
    df_ECINN_FD = pd.read_csv(ECINN_FD_files[index])

    df_exp_sampled = exp_flux_sampling(time_array,df_exp,FullScanT,PortionAnalyzed=PortionAnalyzed)
    df_Conventional_FD_sampled = exp_flux_sampling(time_array,df_Conventional_FD ,FullScanT,PortionAnalyzed=PortionAnalyzed)
    df_ECINN_FD_sampled = exp_flux_sampling(time_array,df_ECINN_FD,FullScanT,PortionAnalyzed=PortionAnalyzed)
    MSE_Conventional = mean_squared_error(df_exp_sampled,df_Conventional_FD_sampled)
    MSE_ECINN = mean_squared_error(df_exp_sampled,df_ECINN_FD_sampled)

    df_exp = df_exp.iloc[:int(PortionAnalyzed*len(df_exp))]
    df_ECINN_FD = df_ECINN_FD[:int(PortionAnalyzed*len(df_ECINN_FD))]
    df_Conventional_FD = df_Conventional_FD[:int(PortionAnalyzed*len(df_Conventional_FD))]    
    ax = axs[0]
    ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=2,label=f'$\\nu=${scan_rates[index]:.2f} V/s, MSE={MSE_Conventional:.3f}',color=tuple(colors[index]))
    ax.plot(df_Conventional_FD .iloc[:,0],df_Conventional_FD .iloc[:,1],ls='--',lw=2,color=tuple(colors[index]))

    ax = axs[1]
    ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=2,label=f'$\\nu=${scan_rates[index]:.2f} V/s, MSE={MSE_ECINN:.3f}',color=tuple(colors[index]))
    ax.plot(df_ECINN_FD.iloc[:,0],df_ECINN_FD.iloc[:,1],ls='--',lw=2,color=tuple(colors[index]))


    

ax = axs[0]
ax.legend()
ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')
ax.set_title('Conventional Method')

ax = axs[1]
ax.legend()
ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')
ax.set_title('ECINN-BV')

fig.text(0.05,0.9,'a)',fontsize=20,fontweight='bold')
fig.text(0.05,0.46,'b)',fontsize=20,fontweight='bold')


fig.savefig("Conventional Method ECINN FD.png",dpi=250,bbox_inches='tight')
