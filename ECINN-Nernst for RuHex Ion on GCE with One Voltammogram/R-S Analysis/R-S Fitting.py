import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from helper import flux_sampling
from sklearn.linear_model import LinearRegression
from sympy.solvers import solve
from sympy import Symbol


linewidth = 4
fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

Excel_File_Name = "RuHex Experiment Jake Aug 7.xlsx"
sheet_names = ['25 mV s-1','50 mV s-1','100 mV s-1','200 mV s-1']
scan_rates = [0.025,0.05,0.1,0.2]
forward_scan_peak_currents = []





# number of test samples
num_test_samples = 1000

r_e = 1.50e-3 #m 
c_bulk = 0.96 # mol per cubic meter
E_i = 0.1 # V vs SCE
E_v = -0.4 # V vs SCEnn

R = 8.314 # Gas constant J/(mol K)
T = 298 # K
F = 96485 # C/mol

A = np.pi* r_e**2 # Area of electrode, m^2 


def solve_RS(coeff):
    D = np.square(coeff/(0.4463*F*A*c_bulk*np.sqrt(F/(R*T))))
    return D

    


fig,ax = plt.subplots(figsize=(8,4.5),dpi=250)

for nu,sheet_name in zip(scan_rates,sheet_names):
    df_exp = pd.read_excel(Excel_File_Name,sheet_name=sheet_name)

    ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=3,label =f'{nu:.3f}V/s')
    peak_current = df_exp.iloc[:,1].min()
    forward_scan_peak_currents.append(peak_current)

    



ax.set_xlabel(r'Potential, V vs. SCE')
ax.set_ylabel(r'Current, A')
ax.legend()

fig.savefig('Experimental.png',dpi=250,bbox_inches='tight')



fig,ax = plt.subplots(figsize=(8,4.5),dpi=250)
lr = LinearRegression()
forward_scan_peak_currents  = np.fabs(forward_scan_peak_currents)

lr.fit(np.sqrt(scan_rates).reshape(-1,1),forward_scan_peak_currents)
coeff = lr.coef_[0]
R2 = lr.score(np.sqrt(scan_rates).reshape(-1,1),forward_scan_peak_currents)

ax.scatter(np.sqrt(scan_rates),forward_scan_peak_currents,marker='o',label='experimental results')

sqrt_scan_rates = np.sqrt(scan_rates)
sqrt_scan_rates = np.append(np.array([[-0.05]]),sqrt_scan_rates)
ax.plot(np.array(sqrt_scan_rates).reshape(-1,1),lr.predict(np.array(sqrt_scan_rates).reshape(-1,1)),ls='--',label='linear regression')
solved_D = solve_RS(coeff=coeff)
ax.set_title(f'$\sqrt{{\\nu}}$ Vs. Current\n slope = {coeff:.4E}\n$R^{{2}}= {R2:.5f}$\n$D calculated = {solved_D:.2E}m^2/s$')
ax.grid(True)
ax.set_xlabel(f'$\sqrt{{\\nu,V/s}}$')
ax.set_ylabel('Current, A')


ax.legend()
fig.savefig('GC electrode fitting.png',dpi=250,bbox_inches = 'tight')
    