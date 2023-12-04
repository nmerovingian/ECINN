import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dBs = [1e-2]
files = ['Data5\sigma=4.00E+03 K0=1.00E+03 alpha=5.00E-01 kinetics=BV mode=linear,dA=1.00E+00,dB=1.00E+00.csv']

def findFormalPotential(df:pd.DataFrame):
    df_forward = df.iloc[:int(len(df)/2)]
    df_reverse = df.iloc[int(len(df)/2):]
    df_reverse = df_reverse.reset_index(drop=True)

    forward_scan_peak_flux = df_forward.iloc[:,1].min()
    print(forward_scan_peak_flux)
    reverse_scan_peak_flux = df_reverse.iloc[:,1].max()
    forward_scan_peak_potential = df_forward.iloc[df_forward.iloc[:,1].idxmin(),0]
    reverse_scan_peak_potential = df_reverse.iloc[df_reverse.iloc[:,1].idxmax(),0]
    E0f = (forward_scan_peak_potential+reverse_scan_peak_potential)/2
    return E0f


for file  in files:
    df = pd.read_csv(file)

    E0f = findFormalPotential(df)
    print('Formal Potential',E0f)

