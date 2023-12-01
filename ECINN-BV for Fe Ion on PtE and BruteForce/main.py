import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from lib.ecinn import ECINN
from lib.network import Network
import pandas as pd
import math
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0: All messages are logged # Default behavior #1: INFO message not printed #2 Warning message not printed #3 INFO,WARNING AND ERROR message not printed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from helper import exp_flux_sampling,toDimensionalPotential,toDimensionlessPotential,expParameters
import itertools
from sklearn.metrics import mean_absolute_error
from ConvertToDimensionless import ERef, r_e,Dref,c_bulk,R,T,F,A

linewidth = 4
fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


outerBoundary = 'SI' # Mode of Outerboudary for simulation. Either SI for Semi-Infinite or TL for Thin-Layer. 




# build a core network model
network1 = Network.build(name='network1')
network2 = Network.build(name='network2')
network3 = Network.build(name='network3')
#network1.summary()
# build a PINN model
ecinn = ECINN(network1,network2,network3).build(outerBoundary)

#tf.keras.utils.plot_model(pinn, to_file='model.png', show_shapes=True,expand_nested=True,dpi=200) #Requires installation of Graphviz and pydot

default_weight_name = "./weights/default.h5"
ecinn.save_weights(default_weight_name)
#print(pinn.trainable_variables) Inspect the names of trainable weights
ecinn.summary()

def get_Lambda_values():
    lambda_K0_value = [v for v in ecinn.trainable_variables if 'lambda_K0' in v.name][0]
    lambda_alpha_value = [v for v in ecinn.trainable_variables if 'lambda_alpha' in v.name][0]
    lambda_beta_value = [v for v in ecinn.trainable_variables if 'lambda_beta' in v.name][0]
    lambda_dA_value = [v for v in ecinn.trainable_variables if 'lambda_d_A' in v.name][0]
    return lambda_K0_value.numpy(),lambda_alpha_value.numpy(),lambda_beta_value.numpy(),lambda_dA_value.numpy()


class AdaptiveWeightCallBack(tf.keras.callbacks.Callback):
    def __init__(self,weights_list):
        super().__init__()

        self.BV_weight_1 = weights_list[6]
        self.BV_weight_2 = weights_list[7]
        self.BV_weight_3 = weights_list[8]

        self.BV_weight_1_hist = []
        self.BV_weight_2_hist = []
        self.BV_weight_3_hist = []
        self.lambda_K0_value_hist = []
        self.lambda_alpha_value_hist = []
        self.lambda_beta_value_hist = []
        self.lambda_dA_value_hist = []
        self.history_dict = dict()




    def on_epoch_end(self, epoch, logs=None):
        


        if epoch>20 and epoch<41:
            self.BV_weight_1.assign_add(0.05)
            self.BV_weight_2.assign_add(0.05)
            self.BV_weight_3.assign_add(0.05)
        
        """
        if epoch>15 and epoch<19:
            self.BV_weight.assign_add(0.5)
        """ 

        print('loss weighting values',self.BV_weight_1.numpy(),self.BV_weight_2.numpy(),self.BV_weight_3.numpy())
        lambda_K0_value,lambda_alpha_value,lambda_beta_value,lambda_dA_value = get_Lambda_values()
        print('K0',lambda_K0_value,'alpha',lambda_alpha_value,'beta',lambda_beta_value,'dA',lambda_dA_value)

        self.BV_weight_1_hist.append(self.BV_weight_1.numpy())
        self.BV_weight_2_hist.append(self.BV_weight_2.numpy())
        self.BV_weight_3_hist.append(self.BV_weight_3.numpy())


        self.lambda_K0_value_hist.append(lambda_K0_value)
        self.lambda_alpha_value_hist.append(lambda_alpha_value)
        self.lambda_beta_value_hist.append(lambda_beta_value)
        self.lambda_dA_value_hist.append(lambda_dA_value)


    def on_train_end(self,logs=None):
        self.history_dict['BV_weight_1'] = self.BV_weight_1_hist
        self.history_dict['BV_weight_2'] = self.BV_weight_2_hist
        self.history_dict['BV_weight_3'] = self.BV_weight_3_hist

        self.history_dict['lambda_K0'] = self.lambda_K0_value_hist
        self.history_dict['lambda_alpha'] = self.lambda_alpha_value_hist
        self.history_dict['lambda_beta'] =  self.lambda_beta_value_hist
        self.history_dict['lambda_dA'] = self.lambda_dA_value_hist

        self.history_dict 

        return self.history_dict 

    







class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,exp_file_names,sigmas,theta_i,theta_v,PortionAnalyzed=1.0,Lambda=3.0,num_train_samples=1000000,batch_size=250):
        super().__init__()
        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.Lambda = Lambda
        self.PortionAnalyzed  = PortionAnalyzed
        self.theta_i = theta_i
        self.theta_v = theta_v

        self.exp_file_name_1 = exp_file_names[0]
        self.exp_file_name_2 = exp_file_names[1]
        self.exp_file_name_3 = exp_file_names[2]

        #Read Experimental Data
        self.df_exp_1 = pd.read_csv(self.exp_file_name_1)
        self.df_exp_2 = pd.read_csv(self.exp_file_name_2)
        self.df_exp_3 = pd.read_csv(self.exp_file_name_3)

        self.sigma_1 = sigmas[0]
        self.sigma_2 = sigmas[1]
        self.sigma_3 = sigmas[2]

        self.peak_flux_1 = np.fabs(self.df_exp_1.iloc[:,1].min())
        self.peak_flux_2 = np.fabs(self.df_exp_2.iloc[:,1].min())
        self.peak_flux_3 = np.fabs(self.df_exp_3.iloc[:,1].min())

        self.peak_potential_1 = np.fabs(self.df_exp_1.iloc[self.df_exp_1.iloc[:,1].idxmin(),0])
        self.peak_potential_2 = np.fabs(self.df_exp_2.iloc[self.df_exp_2.iloc[:,1].idxmin(),0])
        self.peak_potential_3 = np.fabs(self.df_exp_3.iloc[self.df_exp_3.iloc[:,1].idxmin(),0])


        self.FullScanT_1 =  2.0*abs(theta_v-theta_i)/self.sigma_1
        self.FullScanT_2 =  2.0*abs(theta_v-theta_i)/self.sigma_2
        self.FullScanT_3 =  2.0*abs(theta_v-theta_i)/self.sigma_3


        self.maxT_1 = self.FullScanT_1*PortionAnalyzed 
        self.maxT_2 = self.FullScanT_2*PortionAnalyzed 
        self.maxT_3 = self.FullScanT_3*PortionAnalyzed

        self.time_array_1 = np.linspace(0,self.maxT_1,num=self.num_train_samples)
        self.time_array_2 = np.linspace(0,self.maxT_2,num=self.num_train_samples)
        self.time_array_3 = np.linspace(0,self.maxT_3,num=self.num_train_samples)
        np.random.shuffle(self.time_array_1)
        np.random.shuffle(self.time_array_2)
        np.random.shuffle(self.time_array_3)

        self.maxX_1 = Lambda * np.sqrt(self.maxT_1)
        self.maxX_2 = Lambda * np.sqrt(self.maxT_2)  
        self.maxX_3 = Lambda * np.sqrt(self.maxT_3)   


    def getFullScanT(self):
        return [self.FullScanT_1, self.FullScanT_2, self.FullScanT_3]
    
    def getMaxT(self):
        return [self.maxT_1, self.maxT_2, self.maxT_3]
    
    def getMaxX(self):
        return [self.maxX_1,self.maxX_2,self.maxX_3]
    
    def getDf_exp(self):
        return [self.df_exp_1,self.df_exp_2,self.df_exp_3]

    def getPeakFlux(self):
        return [self.peak_flux_1,self.peak_flux_2,self.peak_flux_3]
    
    def getPeakPotential(self):
        return [self.peak_potential_1,self.peak_potential_2,self.peak_potential_3]


    def __len__(self):
        return int(np.floor(self.num_train_samples / self.batch_size))



    def __getitem__(self, index):
    
        # create training input
        #The diffusion equation domain
            #The diffusion equation domain
        TX_eqn_1 = np.random.rand(self.batch_size, 2)
        TX_eqn_1[...,0] = TX_eqn_1[...,0] * self.maxT_1
        TX_eqn_1[..., 1] = TX_eqn_1[..., 1] * self.maxX_1

        TX_eqn_2 = np.random.rand(self.batch_size, 2)
        TX_eqn_2[...,0] = TX_eqn_2[...,0] * self.maxT_2
        TX_eqn_2[..., 1] = TX_eqn_2[..., 1] * self.maxX_2


        TX_eqn_3 = np.random.rand(self.batch_size, 2)
        TX_eqn_3[...,0] = TX_eqn_3[...,0] * self.maxT_3
        TX_eqn_3[..., 1] = TX_eqn_3[..., 1] * self.maxX_3

        # Initial condition T=0
        TX_ini_1 = np.random.rand(self.batch_size, 2)  
        TX_ini_1[..., 0] = 0.0
        TX_ini_1[...,1] = TX_ini_1[...,1]*self.maxX_1      


        TX_ini_2 = np.random.rand(self.batch_size, 2)  
        TX_ini_2[..., 0] = 0.0
        TX_ini_2[...,1] = TX_ini_2[...,1]*self.maxX_2     


        TX_ini_3 = np.random.rand(self.batch_size, 2)  
        TX_ini_3[..., 0] = 0.0
        TX_ini_3[...,1] = TX_ini_3[...,1]*self.maxX_3



        #Boundary condition at electrode surface (X=0)
        TX_bnd0_1 = np.random.rand(self.batch_size, 2)         
        TX_bnd0_1[...,0] = np.sort(self.time_array_1[index*self.batch_size:(index+1)*self.batch_size])
        TX_bnd0_1[..., 1] =  0.0
        
        TX_bnd0_2 = np.random.rand(self.batch_size, 2)         
        TX_bnd0_2[...,0] = np.sort(self.time_array_2[index*self.batch_size:(index+1)*self.batch_size])
        TX_bnd0_2[..., 1] =  0.0

        TX_bnd0_3 = np.random.rand(self.batch_size, 2)         
        TX_bnd0_3[...,0] = np.sort(self.time_array_3[index*self.batch_size:(index+1)*self.batch_size])
        TX_bnd0_3[..., 1] =  0.0



        #Applied Potential
        theta_aux_1 = np.random.rand(self.batch_size, 1)
        theta_aux_2 = np.random.rand(self.batch_size, 1)
        theta_aux_3 = np.random.rand(self.batch_size, 1)
        theta_aux_1[:,0] = np.where(TX_bnd0_1[...,0]<self.FullScanT_1/2.0,self.theta_i-self.sigma_1*TX_bnd0_1[...,0],self.theta_v+self.sigma_1*(TX_bnd0_1[...,0]-self.FullScanT_1/2.0))   
        theta_aux_2[:,0] = np.where(TX_bnd0_2[...,0]<self.FullScanT_2/2.0,self.theta_i-self.sigma_2*TX_bnd0_2[...,0],self.theta_v+self.sigma_2*(TX_bnd0_2[...,0]-self.FullScanT_2/2.0))   
        theta_aux_3[:,0] = np.where(TX_bnd0_3[...,0]<self.FullScanT_3/2.0,self.theta_i-self.sigma_3*TX_bnd0_3[...,0],self.theta_v+self.sigma_3*(TX_bnd0_3[...,0]-self.FullScanT_3/2.0))   

        interpolated_flux_1 = exp_flux_sampling(TX_bnd0_1[...,0],self.df_exp_1,self.FullScanT_1,self.PortionAnalyzed)
        interpolated_flux_2 = exp_flux_sampling(TX_bnd0_2[...,0],self.df_exp_2,self.FullScanT_2,self.PortionAnalyzed)
        interpolated_flux_3 = exp_flux_sampling(TX_bnd0_3[...,0],self.df_exp_3,self.FullScanT_3,self.PortionAnalyzed)
        

        

        #Boundary condition at the outerboundary of simulatuion 
        TX_bnd1_1 = np.random.rand(self.batch_size, 2)
        TX_bnd1_1[...,0] =  TX_bnd1_1[...,0] * self.maxT_1
        TX_bnd1_1[...,1] = self.maxX_1

        TX_bnd1_2 = np.random.rand(self.batch_size, 2)
        TX_bnd1_2[...,0] =  TX_bnd1_2[...,0] * self.maxT_2
        TX_bnd1_2[...,1] = self.maxX_2

        TX_bnd1_3 = np.random.rand(self.batch_size, 2)
        TX_bnd1_3[...,0] =  TX_bnd1_3[...,0] * self.maxT_3
        TX_bnd1_3[...,1] = self.maxX_3


        # create training output
        #Governing equation target
        C_eqn_1 = np.zeros((self.batch_size, 1))
        C_eqn_2 = np.zeros((self.batch_size, 1))
        C_eqn_3 = np.zeros((self.batch_size, 1))
        #Initial equation target              
        C_ini_1 = np.ones((self.batch_size,1))
        C_ini_2 = np.ones((self.batch_size,1))
        C_ini_3 = np.ones((self.batch_size,1))
        #Electrode Surface target    
        C_BV_bnd0_1 = np.zeros((self.batch_size,1))
        C_BV_bnd0_2 = np.zeros((self.batch_size,1))
        C_BV_bnd0_3 = np.zeros((self.batch_size,1))

        #Outerboundary target
        if outerBoundary == 'SI':
            C_bnd1_1 = np.ones((self.batch_size,1))
            C_bnd1_2 = np.ones((self.batch_size,1))
            C_bnd1_3 = np.ones((self.batch_size,1)) #Fixed Concentration Boundary Condition
        elif outerBoundary =='TL': 
            C_bnd1_1 = np.zeros((self.batch_size,1))
            C_bnd1_2 = np.zeros((self.batch_size,1))
            C_bnd1_3 = np.zeros((self.batch_size,1)) #No flux boundary condition

        else:
            raise ValueError



        x_train = [TX_eqn_1, TX_ini_1, TX_bnd0_1,TX_bnd1_1,theta_aux_1,TX_eqn_2, TX_ini_2, TX_bnd0_2, TX_bnd1_2, theta_aux_2, TX_eqn_3, TX_ini_3, TX_bnd0_3, TX_bnd1_3, theta_aux_3]



        y_train = [C_eqn_1,C_eqn_2,C_eqn_3,C_ini_1,C_ini_2,C_ini_3,C_BV_bnd0_1,C_BV_bnd0_2,C_BV_bnd0_3,interpolated_flux_1,interpolated_flux_2,interpolated_flux_3,C_bnd1_1,C_bnd1_2,C_bnd1_3]
        

        return x_train,y_train





def PINNFitExp(exp_file_names,sigmas,theta_i,theta_v,dA_exp_measured,epochs=200,PortionAnalyzed=1.0,Lambda=3.0,initial_weights=None,train=True,saving_directory='./Data'):
    """
    PortionAnalyzed=0.75: We only analyze the firt 75% of voltammogram, only cathodic reaction happens
    
    """
    # saving directory is where data(voltammogram, concentration profile etc is saved)
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    #weights folder is where weights is saved
    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    def schedule(epoch,lr):
        # a learning rate scheduler 
        if epoch<=50:
            return lr 
        else:
            lr *= 0.98
            return lr


    # number of training samples
    num_train_samples = 1000000
    # number of test samples
    num_test_samples = 1000

    batch_size = 250



    file_name = f'sigmas=[{sigmas[0]:.2E},{sigmas[1]:.2E},{sigmas[2]:.2E}] epochs={epochs} outerBnd={outerBoundary} Lambda={Lambda:.2f} epochs = {epochs:.2E} n_train = {num_train_samples:.2E} dA_exp_measured={dA_exp_measured}'

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

    # The initial weights regarding to bnd0 (BV) boundary condition at electrode surface is set to 0 for a few epochs. Then slowly increased to 1 latter.
    weights_list = [1.0,1.0,1.0,1.0,1.0,1.0,tf.Variable(0.0,trainable=False),tf.Variable(0.0,trainable=False),tf.Variable(0.0,trainable=False),1.0,1.0,1.0,1.0,1.0,1.0]
    adptiveweightcallback = AdaptiveWeightCallBack(weights_list)

    ecinn.compile(optimizer='adam',loss='mse',loss_weights = weights_list)
    ecinn.load_weights(default_weight_name)


    training_generator = DataGenerator(exp_file_names,sigmas,theta_i,theta_v,PortionAnalyzed,Lambda,num_train_samples,batch_size)


    if train:
        if initial_weights:
            ecinn.load_weights(initial_weights)
        history = ecinn.fit(training_generator,epochs=epochs,verbose=2,callbacks=[lr_scheduler_callback,adptiveweightcallback],workers=3,use_multiprocessing=False,max_queue_size=24)
        history_callback = adptiveweightcallback.history_dict
        df_history = pd.DataFrame({**history.history,**history_callback})
        df_history.to_csv(f'{saving_directory}/history {file_name}.csv')
        ecinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            ecinn.load_weights(f'./weights/weights {file_name}.h5')
            ecinn.save_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            if initial_weights:
                ecinn.load_weights(initial_weights)

            history = ecinn.fit(training_generator,epochs=epochs,verbose=2,callbacks=[lr_scheduler_callback,adptiveweightcallback],workers=3,use_multiprocessing=False,max_queue_size=24)
            history_callback = adptiveweightcallback.history_dict
            df_history = pd.DataFrame({**history.history,**history_callback})
            df_history.to_csv(f'{saving_directory}/history {file_name}.csv')
            ecinn.save_weights(f'./weights/weights {file_name}.h5')


    #Get the trained variable for the inverse problem
    lambda_K0_value,lambda_alpha_value,lambda_beta_value,lambda_dA_value = get_Lambda_values()

    networks = [network1,network2,network3]
    maxTs = training_generator.getMaxT()
    FullScanTs = training_generator.getFullScanT()
    maxXs = training_generator.getMaxX()
    df_exps = training_generator.getDf_exp()
    peak_fluxes = training_generator.getPeakFlux()
    peak_potentials = training_generator.getPeakPotential()


    # plot u(t,x) distribution as a color-map
    fig_all,axs_all = plt.subplots(figsize=(24,9),nrows=2,ncols=3)
    colors = cm.viridis(np.linspace(0,1,len(networks)))
    for i in range(len(networks)):

        network = networks[i]
        maxT = maxTs[i]
        FullScanT = FullScanTs[i]
        maxX = maxXs[i]
        sigma = sigmas[i]
        scan_rate =scan_rates[i]

        df_exp = df_exps[i]
        peak_flux_exp = peak_fluxes[i]
        peak_potential_exp = peak_potentials[i]


        # predict c(t,x) distribution
        t_flat = np.linspace(0, maxT, num_test_samples)
        cv_flat = np.where(t_flat<FullScanT/2.0,theta_i-sigma*t_flat,theta_v+sigma*(t_flat-FullScanT/2.0))
        x_flat = np.linspace(0, maxX, num_test_samples) 
        t, x = np.meshgrid(t_flat, x_flat)
        tx = np.stack([t.flatten(), x.flatten()], axis=-1)
        c = network.predict(tx, batch_size=num_test_samples)
        c = c.reshape(t.shape)

        exp_flux = exp_flux_sampling(t_flat,df_exp,FullScanT,PortionAnalyzed)

        TX_flux = np.zeros((len(t_flat),2))
        TX_flux[:,0] = t_flat

        TX_flux = tf.convert_to_tensor(TX_flux,dtype=tf.float32)

        with tf.GradientTape() as g:
            g.watch(TX_flux)
            C = network(TX_flux)

        dC_dX = g.batch_jacobian(C,TX_flux)[...,1]
        flux = -lambda_dA_value* dC_dX.numpy().reshape(-1)
        peak_flux_pinn = np.fabs(np.min(flux))
        peak_potential_pinn = np.fabs(cv_flat[np.argmin(flux)])

        flux_mae = mean_absolute_error(exp_flux,flux)
        flux_mae_max = np.max(np.fabs(exp_flux-flux))

        df = pd.DataFrame({'Potential':cv_flat,'Flux':flux})
        df.to_csv(f'{saving_directory}/PINN {file_name} {i}.csv',index=False)

        fig,ax = plt.subplots(figsize=(8,4.5))

        ax.plot(cv_flat,flux,label='PINN,')
        #ax.axhline(-0.446*math.sqrt(sigma),label='R-S equation',ls='--',color='r')
        ax.axvline(-1.109,label='Expected Forward Scan Potential',ls='--',color='k')
        ax.set_xlabel('Potential, theta')
        ax.set_ylabel('Flux,J')
        ax.legend()

        fig.savefig(f'{saving_directory}/CV {file_name} {i}.png')
    




        ax = axs_all[0][i]
        axes = ax.pcolormesh(t, x, c, cmap='rainbow',shading='auto')

        ax.set_xlabel('T',fontsize='large',fontweight='bold')
        ax.set_ylabel('X',fontsize='large',fontweight='bold')
        #ax.set_ylim(0,maxX*0.4)
        cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)
        cbar.set_label('C(T,X)',fontsize='large',fontweight='bold')
        axes.set_clim(0.0, 1)

        ax = axs_all[1][i]
        df.plot(x='Potential',y='Flux',ax=ax,color='b',lw=3,alpha=0.7,label='PINN')


        df_exp.plot(x=0,y=1,ax=ax,color='r',ls='--',lw=3,alpha=0.7,label='Experiment')
        #ax.plot([-5,5],[-0.446*math.sqrt(sigma),-0.446*math.sqrt(sigma)],label='R-S Equation',ls='--',color='k',lw=3)
        ax.annotate("", xy=(2.5, -0.5), xytext=(10, -0.5),arrowprops=dict(facecolor='black', shrink=0.05))
        ax.set_xlabel(r'Potential,$\theta$')
        ax.set_ylabel(r'Flux, $J$')
        ax.set_title(f'$\\nu={scan_rates[i]:.2E}$V/s,$\sigma={sigmas[i]:.2E}$\n$J_{{mae}}/J_{{peak}}$={flux_mae/peak_flux_exp:.3%} $J_{{maxae}}/J_{{peak}}$={flux_mae_max/peak_flux_exp:.3%}\n$\epsilon J_{{peak}}=$={np.fabs((peak_flux_pinn-peak_flux_exp)/peak_flux_exp):.3%} $\epsilon \\theta_{{peak}}$ = {np.fabs((peak_potential_pinn-peak_potential_exp)/peak_potential_exp):.3%}')
        sec_ax = ax.secondary_xaxis(-0.15,functions=(toDimensionalPotential,toDimensionlessPotential))
        sec_ax.set_xlabel(f'$E-E^0_f$, V')

        ax.legend(fontsize=15)

    from ConvertToDimensionless import ERef, r_e,Dref
    lambda_k0 = lambda_K0_value*Dref/r_e
    lambda_k0_eff_c = lambda_k0*math.exp(lambda_alpha_value*ERef*96485/8.314/298)
    lambda_k0_eff_a = lambda_k0*math.exp(-lambda_beta_value*ERef*96485/8.314/298)



    fig_all.suptitle(f'$\lambda_{{K_0}}$={lambda_K0_value:.2E},$\lambda_{{k_0}}$={lambda_K0_value*1e-9/1.5e-3:.2E}m/s\n $\lambda_{{k_0,eff,c}}$={lambda_k0_eff_c:.2E} m/s $\lambda_{{k_0,eff,a}}$={lambda_k0_eff_a:.2E} m/s\n $\lambda_{{\\alpha}}$ = {lambda_alpha_value:.2f},$\lambda_{{\\beta}}$ = {lambda_beta_value:.2f}, $\lambda_{{d_A}}$ = {lambda_dA_value:.3f}, $\lambda_{{D_A}}={lambda_dA_value*1e-9:.2E}m^2s^{{-1}}$',y=1.1)
    fig_all.tight_layout()
    fig_all.savefig(f'{saving_directory}/PINN {file_name}.png',dpi=250,bbox_inches='tight')
    
    plt.close('all')
    #Plot in dimensional forms for paper illustration
    fig = plt.figure(figsize=(8,9),)
    gs = fig.add_gridspec(2,3,width_ratios=[5,2,1],wspace=0.08)
    ax_cv = fig.add_subplot(gs[1,:])

    for i in range(len(networks)):
        network = networks[i]
        maxT = maxTs[i]
        FullScanT = FullScanTs[i]
        maxX = maxXs[i]
        sigma = sigmas[i]
        scan_rate =scan_rates[i]
        # predict c(t,x) distribution
        t_flat = np.linspace(0, maxT, num_test_samples)
        cv_flat = np.where(t_flat<FullScanT/2.0,theta_i-sigma*t_flat,theta_v+sigma*(t_flat-FullScanT/2.0))
        x_flat = np.linspace(0, maxX, num_test_samples) 
        t, x = np.meshgrid(t_flat, x_flat)
        tx = np.stack([t.flatten(), x.flatten()], axis=-1)
        c = network.predict(tx, batch_size=num_test_samples)
        c = c.reshape(t.shape)


        TX_flux = np.zeros((len(t_flat),2))
        TX_flux[:,0] = t_flat

        TX_flux = tf.convert_to_tensor(TX_flux,dtype=tf.float32)

        with tf.GradientTape() as g:
            g.watch(TX_flux)
            C = network(TX_flux)

        dC_dX = g.batch_jacobian(C,TX_flux)[...,1]
        flux = -lambda_dA_value* dC_dX.numpy().reshape(-1)
        
        df = pd.DataFrame({'Potential':cv_flat,'Flux':flux})
        df.to_csv(f'{saving_directory}/PINN {file_name} {i}.csv',index=False)

        df_exp = df_exps[i]


        ax = fig.add_subplot(gs[0,i])
        
        #convert to dimensional
        t_dim = t*r_e*r_e/Dref
        x_dim = x*r_e * 1e3 # Change units to mm
        c_dim = c*c_bulk
        axes = ax.pcolormesh(t_dim, x_dim, c_dim, cmap='YlGnBu',shading='auto')
        ax.set_ylim(0,0.5)


        ax.set_xlabel('$t / s$',fontsize='large',fontweight='bold')

        if i==0:
            ax.set_ylabel('$x / mm$',fontsize='large',fontweight='bold')

        if i==1 or i==2:
            ax.yaxis.set_tick_params(labelleft=False)
            ax.set_yticks([])
        if i==2:
            cbar = plt.colorbar(axes,pad=0.15, aspect=28,ax=ax)
            cbar.set_label(r'$c_{Fe^{3+}}(t,x)/mM$',fontsize='large',fontweight='bold')
            axes.set_clim(0.0, 1.0*c_bulk+0.1)

        ax = ax_cv

        df['Potential'] = df['Potential'] / (F/(R*T)) + ERef
        df['Flux'] = df['Flux'] * (F*A*c_bulk*Dref/r_e) * 1e6 # Change unit to micronA
        df.plot(x='Potential',y='Flux',ax=ax,color=tuple(colors[i]),lw=3,alpha=0.7,label=f'$\\nu={scan_rates[i]:.2f}$V/s')

        df_exp = df_exp.iloc[:int(PortionAnalyzed*len(df_exp))]
        df_exp.iloc[:,0] = df_exp.iloc[:,0]/ (F/(R*T)) + ERef
        df_exp.iloc[:,1] = df_exp.iloc[:,1] * (F*A*c_bulk*Dref/r_e)* 1e6 # Change unit to micronA
        ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],color=tuple(colors[i]),ls='--',lw=3,alpha=0.7)
        #ax.plot([-5,5],[-0.446*math.sqrt(sigma),-0.446*math.sqrt(sigma)],label='R-S Equation',ls='--',color='k',lw=3)
        ax.annotate("", xy=(0.6, -3), xytext=(0.75, -3),arrowprops=dict(facecolor='black', shrink=0.05))
        ax.set_ylabel(r'$I/\mu A$',fontsize="large",fontweight='bold')
        ax.set_xlabel(r'E/V vs. SCE',fontsize='large',fontweight='bold')


        ax.legend(fontsize=16)

    fig.text(0.05,0.90,'a)',fontsize=20,fontweight='bold')
    fig.text(0.05,0.48,'b)',fontsize=20,fontweight='bold')

    fig.savefig(f'{saving_directory}/PINN Paper {file_name} {i}.png',dpi=250,bbox_inches='tight')



    return f'./weights/weights {file_name}.h5'



if __name__ == "__main__":

    scan_rates = [0.01,0.02,0.05]
    exp_dimensionless_files = [
        "ExpData/Exp Dimensionless sigma=2.8137E+02 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
        "ExpData/Exp Dimensionless sigma=5.6273E+02 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",
        "ExpData/Exp Dimensionless sigma=1.4068E+03 theta_i=1.4269E+01 theta_v=-1.2992E+01 dA=4.63E-01.csv",]


    sigmas =[]
    for exp_dimensionless_file in exp_dimensionless_files:
        sigma,theta_i,theta_v,dA = expParameters(exp_dimensionless_file)
        sigmas.append(sigma)

    for epochs in [300]:
        saving_directory = f"Paper Figure Epochs={epochs}"
        PINNFitExp(exp_dimensionless_files,sigmas,theta_i,theta_v,dA,epochs=epochs,train=False,saving_directory=saving_directory)

    