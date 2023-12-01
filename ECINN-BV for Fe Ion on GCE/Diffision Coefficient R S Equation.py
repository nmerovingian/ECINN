import pandas as pd
from matplotlib import pyplot as plt
import numpy as np 
from helper import exp_flux_sampling,toDimensionalPotential,toDimensionlessPotential,expParameters
from matplotlib import cm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Now you have at your disposition several error estimates, e.g.
"""
Estimate the diffusion coeffcient of Fe3+ using irreversible Randles-Sevick equation 

"""

alpha = 0.38 # Using the transfer coefficient obtained from "Tafel Analysis Tafel Region Method.py"


linewidth = 3
fontsize = 14
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


scan_rates = [0.01,0.02,0.05,0.1,0.2] # V/s 
exp_dimensionless_files = [
    "ExpData/Exp Dimensionless sigma=8.7623E+02 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=1.7525E+03 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
    "ExpData/Exp Dimensionless sigma=4.3811E+03 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
    "ExpData\Exp Dimensionless sigma=8.7623E+03 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
    "ExpData\Exp Dimensionless sigma=1.7525E+04 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv"
]


sigmas =[]
peak_fluxes = []


for index, exp_dimensionless_file in enumerate(exp_dimensionless_files):
    sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_file)
    sigmas.append(sigma)
    df = pd.read_csv(exp_dimensionless_file)
    peak_flux = df.iloc[:,1].min()
    peak_fluxes.append(abs(peak_flux))


fig, ax = plt.subplots(figsize=(8,4.5))

sigmas_sqrt = np.sqrt(sigmas)
ax.scatter(sigmas_sqrt,peak_fluxes)

lr = LinearRegression()
lr.fit(sigmas_sqrt.reshape(-1,1),peak_fluxes)
ols = sm.OLS(peak_fluxes, sm.add_constant(sigmas_sqrt.reshape(-1,1)))
ols_result = ols.fit()
print(ols_result.summary())
print(ols_result.HC0_se)
print(ols_result.HC1_se)
print(ols_result.HC2_se)
sigmas_sqrt = np.concatenate(([0.0],sigmas_sqrt))
ax.plot(sigmas_sqrt,lr.predict(sigmas_sqrt.reshape(-1,1)),ls='--',label=f'$J_{{peak}}$=({lr.coef_[0]:.3f}$\pm${ols_result.HC0_se[1]:.3f})$\sqrt{{\sigma}}$+({lr.intercept_:.2f}$\pm${ols_result.HC0_se[0]:.2f})')





d_calc = ((lr.coef_[0])/np.sqrt(alpha)/0.496)**2
print('The diffusion coefficient predicted using irreversible R-S equation is',d_calc)



ax.set_ylabel(r'$|J_{peak}|$',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$\sqrt{\sigma}$',fontsize='large',fontweight='bold')
ax.legend()


fig.savefig('Diffusion Coefficient R S Equation.png',dpi=250,bbox_inches='tight')





