import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import cm
from helper import flux_sampling,expParameters
from sklearn.metrics import mean_squared_error
from FD_Simulation_Unequal_D.FD_Simulation import simulation
import itertools
import multiprocessing
import os

linewidth = 4
fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

scan_rates = [0.01,0.02,0.05]
sigmas = [2.8137E+02,5.6273E+02,1.4068E+03]
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


theta_i=1.4269E+01
theta_v=-1.2992E+01


brute_force_folder = 'FD_Brute'
if not os.path.exists(brute_force_folder):
    os.mkdir(brute_force_folder)



    
def run_simulations(command:dict):
    for index in range(len(sigmas)):
        simulation(sigma=sigmas[index],theta_i=theta_i,theta_v=theta_v,**command)


    


def FD_BruteForce_Parameters(num_K0s=50,step_alphas=0.05,step_betas=0.05,num_davgs=11,parameterFileName='BruteForceParameters.csv'):

    K0s = np.logspace(-7,5,num=num_K0s) # The standard electrochemical rate constants parameters to sweep
    alphas = np.arange(0.3,0.7,step=step_alphas) # The cathodic transfer coefficient to sweep 
    betas =  np.arange(0.3,0.7,step=step_betas) # The anoidc transfer coefficient to sweep 
    davgs = np.logspace(-1,0,num=num_davgs) # the average diffusion coefficient to sweep
    
    print(f'There are {len(K0s)} K0, {len(alphas)} alpha, {len(betas)} beta, {len(davgs)} d values')
    
    total_simulations = len(K0s) * len(alphas) * len(betas) * len(davgs) * 3
    print('The total number of FD simulations are:',total_simulations)

    parameters = itertools.product(K0s,alphas,betas,davgs)
    parameters = list(parameters)
    parameters = np.array(parameters)
    df = pd.DataFrame(parameters,columns=['K0','alpha',"beta",'davg'])
    df.to_csv(parameterFileName,index=False)
    return parameterFileName


def multiprocessingFD(parameterFileName):
    df_parameters = pd.read_csv(parameterFileName)

    commands = []
    
    for index in range(len(df_parameters)):

        K0,alpha,beta,davg = df_parameters.iloc[index]
        commands.append({"K0":K0,"alpha":alpha,"beta":beta,"kinetics":'BV',"mode":'linear',"dA":davg,"dB":davg,"saving_directory":f"{brute_force_folder}/Data {index}"})

    
    with multiprocessing.Pool(processes=os.cpu_count()-1) as pool:
        pool.map(run_simulations,commands)


    print('Job Completed')
    return commands






def evaluate(parameterFileName,commands,resultsFileName='BruteForceResults.csv'):
    print('Evaluating Results')
    df_parameters = pd.read_csv(parameterFileName)
    for i in range(len(scan_rates)):
        sigma = sigmas[i]
        maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
        time_array = np.linspace(0,maxT,num=num_test_samples)

        df_parameters[f'MSE {i}'] = 100 # Initialize with an arbitrary large number 
        df_exp = pd.read_csv(exp_dimensionless_files[i])
        df_exp_sampled = flux_sampling(time_array,df_exp,maxT)
        for j in range(len(df_parameters)):
            if j%2000 == 0:
                print(f'J={j}')
            command = commands[j]
            FD_file_name = simulation(sigma=sigma,theta_i=theta_i,theta_v=theta_v,**command)
            df_FD = pd.read_csv(FD_file_name)
            try:
                df_FD_sampled = flux_sampling(time_array,df_FD,maxT)
            except:
                continue
            MSE = mean_squared_error(df_exp_sampled,df_FD_sampled)
            df_parameters[f'MSE {i}'].iloc[j] = MSE

    MSE_names = [f'MSE {i}' for i in range(len(scan_rates))]
    df_parameters['MSEavg'] = df_parameters[MSE_names].mean(axis=1)
    best_index = df_parameters['MSEavg'].idxmin()
    df_parameters = df_parameters.sort_values(by='MSEavg',ascending=True)
    df_parameters.to_csv(resultsFileName,index=False)
    
    return resultsFileName,best_index
             
            

def plotBest(resultsFileName,commands,best_index):
    df_results = pd.read_csv(resultsFileName)
     
    colors = cm.viridis(np.linspace(0,1,len(scan_rates)))
    fig,ax = plt.subplots(figsize=(8,4.5))
    for i in range(len(scan_rates)):
        sigma = sigmas[i]
        maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
        time_array = np.linspace(0,maxT,num=num_test_samples)

        FD_file_name = simulation(sigma=sigma,theta_i=theta_i,theta_v=theta_v,**commands[best_index])
        df_FD = pd.read_csv(FD_file_name)
        df_FD_sampled = flux_sampling(time_array,df_FD,maxT)

        df_exp = pd.read_csv(exp_dimensionless_files[i])
        df_exp_sampled = flux_sampling(time_array,df_exp,maxT)

        MSE =mean_squared_error(df_exp_sampled,df_FD_sampled)


        ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],lw=2,label=f'$\\nu=${scan_rates[i]:.2f} V/s, MSE={MSE:.3f}',color=tuple(colors[i]))
        ax.plot(df_FD.iloc[:,0],df_FD.iloc[:,1],ls='--',lw=2,color=tuple(colors[i]))


    ax.legend()
    ax.set_xlabel(r'Potential, $\theta$',fontsize='large',fontweight='bold')
    ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')

    fig.savefig("FD Best.png",dpi=250,bbox_inches='tight')


if __name__ == "__main__":

    parameterFileName = FD_BruteForce_Parameters()
    
    commands = multiprocessingFD(parameterFileName)
    resultsFileName,best_index = evaluate(parameterFileName,commands)
    print(best_index)
    print(commands[best_index])
    plotBest(resultsFileName,commands,best_index)
    
    
    

