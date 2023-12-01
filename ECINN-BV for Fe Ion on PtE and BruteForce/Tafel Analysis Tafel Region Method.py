import pandas as pd
from matplotlib import pyplot as plt
import numpy as np 
from helper import exp_flux_sampling,toDimensionalPotential,toDimensionlessPotential,expParameters
from matplotlib import cm


linewidth = 3
fontsize = 14
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


scan_rates = [0.01,0.02,0.05,0.1,0.2] # V/s 
exp_dimensionless_files = [
    "ExpData/Exp Dimensionless sigma=2.8137E+02 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=5.6273E+02 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=1.4068E+03 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=2.8137E+03 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=5.6273E+03 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv"]

sigmas =[]
alphas = []


high_end = 0.30
low_end = 0.10
# According to the paper Journal of Electroanalytical Chemistry, 826, 117-124, 2018, the Tafel region is 10% to 30% before peak. 

colors = cm.viridis(np.linspace(0,1,len(scan_rates)))


fig,axs = plt.subplots(figsize=(8,9),nrows=2)
for index, exp_dimensionless_file in enumerate(exp_dimensionless_files):
    sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_file)
    sigmas.append(sigma)
    df = pd.read_csv(exp_dimensionless_file)


    ax = axs[0]
    ax.plot(df.iloc[:,0],df.iloc[:,1],linewidth=linewidth,ls='--',color=tuple(colors[index]),label=f'{scan_rates[index]:.2f} V/s',alpha=0.25)
    peak_flux = df.iloc[:,1].min()
    peak_flux_index = df.iloc[:,1].idxmin()


    df = df[:peak_flux_index]

    df = df[(df.iloc[:,1]<low_end*peak_flux)&(df.iloc[:,1]>high_end*peak_flux)]

    ax.plot(df.iloc[:,0],df.iloc[:,1],linewidth=linewidth,ls='-',color=tuple(colors[index]))

    step = df.iloc[1,0] - df.iloc[0,0] 

    df['Gradient'] = -np.gradient(np.log(-df.iloc[:,1]),step)
    alpha = df['Gradient'].mean()
    alphas.append(alpha)

    ax = axs[1]

    
    df.plot(x='Potential',y='Gradient',ax=ax,linewidth=linewidth,color=tuple(colors[index]),label=f'{scan_rates[index]:.2f} V/s, $\\alpha={alpha:.2f}$',marker='o')


ax = axs[0]
ax.set_xlabel(r'Potential,$\theta$',fontweight = "bold",fontsize='large')
ax.set_ylabel(r'Flux, $J$',fontweight = "bold",fontsize='large')
ax.legend()

ax = axs[1]
ax.tick_params(labelsize='large')
ax.set_xlabel(r"Potential, $\theta$", fontweight = "bold",fontsize='large')
ax.set_ylabel("Apparent transfer coefficient\n$-dln(-J)/d\\theta$", fontweight = "bold",fontsize='large')

fig.text(0.05,0.9,'a)',fontsize=20,fontweight='bold')
fig.text(0.05,0.48,'b)',fontsize=20,fontweight='bold')

fig.savefig('Tafel Analysis Tafel Region Method.png',dpi=250,bbox_inches='tight')

print(f'For all voltammograms alpha ={np.average(alphas)}+/- {np.std(alphas)}')