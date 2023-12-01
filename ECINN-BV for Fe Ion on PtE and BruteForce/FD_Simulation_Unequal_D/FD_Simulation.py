import numpy as np 
from FD_Simulation_Unequal_D.coeff import Coeff
from FD_Simulation_Unequal_D.grid import Grid
import scipy
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import pyplot as plt
import time
import pandas as pd 
from concurrent.futures import ProcessPoolExecutor
import os


bulk_A = 1.0
bulk_B = 0.0



def simulation(sigma=1.0,K0=1.0,alpha=0.5,beta=None,kinetics='BV',mode='linear',dA=1.0,dB=1.0,theta_i=20.0,theta_v = -20.0,saving_directory='./FD_Data'):
    #K0: Standard electrochemical rate constants
    #alpha: Transfer coefficient
    #kinetics: Electrode kinetics, Nernst or BV
    #mode: Mode of mass transport

    if not os.path.exists(saving_directory):
         os.mkdir(saving_directory)

    if beta is None:
        beta = 1.0-alpha

    #start and end potential of voltammetric scan
    cycles = 1
    deltaX = 5e-7 # The initial space step
    omega = 1.08 # Expanding grid factor
    deltaTheta = 2e-2
    deltaT = deltaTheta/sigma
    maxT = cycles*2.0*abs(theta_v-theta_i)/sigma
    if mode == 'linear':
        maxX = 6.0 * np.sqrt(maxT)
        X_grid = [0.0]
        x = 0.0
    elif mode == 'radial':
        maxX = 1.0 +  6.0 * np.sqrt(maxT)
        X_grid = [1.0]
        x = 1.0
    else:
        raise ValueError

    while x <= maxX:
        x += deltaX
        X_grid.append(x)
        deltaX *= omega
    n = len(X_grid)
    X_grid = np.array(X_grid)

        

    

    grid = Grid(n,X_grid,K0,kinetics,alpha,beta,dA,dB)
    grid.init_c(bulk_A,bulk_B)
    coeff = Coeff(n,deltaT,grid.X_grid,K0,alpha,beta,kinetics,mode,dA,dB)
    coeff.Allcalc_abc(deltaT=deltaT)



    #simulation steps
    nTimeSteps = int(2*np.fabs(theta_v-theta_i)/deltaTheta)+1
    Esteps = np.arange(nTimeSteps)
    E = np.where(Esteps<nTimeSteps/2.0,theta_i-deltaTheta*Esteps,theta_v+deltaTheta*(Esteps-nTimeSteps/2.0))
    E = np.tile(E,cycles)
    start_time = time.time()


    CV_location = f'{saving_directory}/sigma={sigma:.2E} K0={K0:.2E} alpha={alpha:.2E} beta={beta:.2E} kinetics={kinetics} mode={mode},dA={dA:.2E},dB={dB:.2E} theta_i={theta_i:.2E} theta_v={theta_v:.2E}'

    if os.path.exists(f'{CV_location}.csv'):
        #print('File Exists')
        return f'{CV_location}.csv'

    fluxes = []
    for index in range(0,int(len(E))):
        Theta = E[index]
        """
        if index == 10:
            print(f'Total run time is {(time.time()-start_time)*len(E)/60/10:.2f} mins')
        """
        coeff.CalcMatrix(Theta)
        grid.update_d(Theta,bulk_A,bulk_B)
        grid.conc = linalg.spsolve(sparse.csr_matrix(coeff.A_matrix),sparse.csr_matrix(grid.conc_d[:,np.newaxis]))
        flux = grid.grad()
        fluxes.append(flux)
        
        """
        if index == 0:
            grid.save_conc_profile(f'{CV_location} conc {index/len(E):.2f}.csv')
        """

    
    df = pd.DataFrame({'Potential':E,'Flux':fluxes})
    df.to_csv(f'{CV_location}.csv',index=False)
    return f'{CV_location}.csv'


        