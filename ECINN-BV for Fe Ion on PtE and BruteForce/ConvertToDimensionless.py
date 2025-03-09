import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from helper import flux_sampling

linewidth = 4
fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


Excel_File_Name = "5 mM Fe(3) 1 M H2SO4 on PtE.xlsx"
sheet_names = ['10 mV s-1','20 mV s-1','50 mV s-1','100 mV s-1','200 mV s-1'][::-1]
scan_rates = [0.01,0.02,0.05,0.1,0.2][::-1]

# number of test samples
num_test_samples = 1000

r_e = 0.85e-3 #m 
c_bulk = 4.85 # mol per cubic meter
E_i = 0.8 # V vs SCE
E_v = 0.1 # V vs SCEnn

R = 8.314 # Gas constant J/(mol K)
T = 298 # K
F = 96485 # C/mol

A = np.pi* r_e**2 # Area of electrode, m^2 

DA = 4.63e-10 #m^2 s^-1

Dref = 1e-9 #m^2 s^-1

ERef = 0.4336 # Formal Potential of Fe2+/Fe3+ couple

exp_saving_directory = './ExpData'
if not os.path.exists(exp_saving_directory):
    os.mkdir(exp_saving_directory)


def findMidPointPotential(df:pd.DataFrame):
    df_forward = df.iloc[:int(len(df)/2)]
    df_reverse = df.iloc[int(len(df)/2):]
    df_reverse = df_reverse.reset_index(drop=True)

    forward_scan_peak_flux = df_forward.iloc[:,1].min()
    reverse_scan_peak_flux = df_reverse.iloc[:,1].max()
    forward_scan_peak_potential = df_forward.iloc[df_forward.iloc[:,1].idxmin(),0]
    reverse_scan_peak_potential = df_reverse.iloc[df_reverse.iloc[:,1].idxmax(),0]
    E0f = (forward_scan_peak_potential+reverse_scan_peak_potential)/2
    return E0f


for nu,sheet_name in zip(scan_rates,sheet_names):
    df_exp = pd.read_excel(Excel_File_Name,sheet_name=sheet_name)

    sigma = (r_e**2/Dref)*(F/(R*T))*nu
    theta_i = (E_i-ERef)* (F/(R*T))
    theta_v = (E_v-ERef)* (F/(R*T))
    dA = DA/Dref
    df_exp.iloc[:,0] = (df_exp.iloc[:,0] - ERef) * (F/(R*T))
    df_exp.iloc[:,1] = (df_exp.iloc[:,1])/(F*A*c_bulk*Dref/r_e)

    df_exp = df_exp.rename({"Potential(V)":"Potential","Current(I)":"Flux"},axis=1)

    df_exp.to_csv(f'{exp_saving_directory}/Exp Dimensionless sigma={sigma:.4E} theta_i={theta_i:.4E} theta_v={theta_v:.4E} dA={dA:.2E}.csv',index=False)


fig,ax = plt.subplots(figsize=(8,4.5))

for nu,sheet_name in zip(scan_rates,sheet_names):
    df_exp = pd.read_excel(Excel_File_Name,sheet_name=sheet_name)
    Emid = findMidPointPotential(df_exp)
    ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=3,label=f'{nu:.3f}V/s,Emid={Emid:.3f}V')



ax.set_xlabel('V vs. SCE')
ax.set_ylabel('A')
ax.legend()
fig.savefig('Experiment.png',dpi=250,bbox_inches='tight')

    
