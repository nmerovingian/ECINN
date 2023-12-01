import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from helper import flux_sampling

Excel_File_Name = "0.96 mM RuHex on GCE.xlsx"
sheet_names = ['25 mV s-1','50 mV s-1','100 mV s-1','200 mV s-1']
scan_rates = [0.025,0.05,0.1,0.2]

# number of test samples
num_test_samples = 1000

r_e = 1.50e-3 #m
rref = 5e-4 # A reference distance
c_bulk = 0.96 # mol per cubic meter
E_i = 0.1 # V vs SCE
E_v = -0.4 # V vs SCEnn

R = 8.314 # Gas constant J/(mol K)
T = 298 # K
F = 96485 # C/mol

A = np.pi*rref**2 # Area of electrode, m^2 

DA = 8.43e-10 #m^2 s^-1
DB = 1.19e-9 #m^2 s^-1

Dref = 1e-9

exp_saving_directory = f'./ExpData Aug 7 rref={rref:.2E}' 

if not os.path.exists(exp_saving_directory):
    os.mkdir(exp_saving_directory)

ERef = -0.17 #V, an initial guess mid point potential


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
    EMid = (forward_scan_peak_potential+reverse_scan_peak_potential)/2
    return EMid


for nu,sheet_name in zip(scan_rates,sheet_names):
    df_exp = pd.read_excel(Excel_File_Name,sheet_name=sheet_name)
    
    #ERef =  findMidPointPotential(df_exp)
    sigma = (rref**2/Dref)*(F/(R*T))*nu
    theta_i = (E_i-ERef)* (F/(R*T))
    theta_v = (E_v-ERef)* (F/(R*T))
    dA = DA/Dref
    dB = DB/Dref
    df_exp.iloc[:,0] = (df_exp.iloc[:,0] - ERef) * (F/(R*T))
    df_exp.iloc[:,1] = (df_exp.iloc[:,1]*((rref*rref/r_e/r_e)))/(F*A*c_bulk*Dref/rref)

    df_exp = df_exp.rename({"Potential(V)":"Potential","Current(A)":"Flux"},axis=1)

    df_exp.to_csv(f'{exp_saving_directory}/Exp Dimensionless sigma={sigma:.4E} theta_i={theta_i:.4E} theta_v={theta_v:.4E} dA={dA:.2E} dB={dB:.2E} ERef={ERef:.3E}.csv',index=False) 

    