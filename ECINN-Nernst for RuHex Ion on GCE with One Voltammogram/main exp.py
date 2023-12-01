import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import math
import os
import tensorflow as tf
from helper import exp_flux_sampling,toDimensionalPotential,toDimensionlessPotential,expParameters
from ConvertToDimensionless import ERef, r_e,Dref,c_bulk,R,T,F,A,rref

STANDARDIZE_FLUX = False

linewidth = 4
fontsize = 12

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


outerBoundary = 'SI' # Mode of Outerboudary for simulation. Either SI for Semi-Infinite or TL for Thin-Layer. 


scan_rates = [0.025,0.05,0.1,0.2][:3]
exp_dimensionless_files = [
    f"ExpData Aug 7 rref={rref:.2E}\Exp Dimensionless sigma=2.4340E+02 theta_i=1.0515E+01 theta_v=-8.9570E+00 dA=8.43E-01 dB=1.19E+00 ERef=-1.700E-01.csv",
    f"ExpData Aug 7 rref={rref:.2E}\Exp Dimensionless sigma=4.8679E+02 theta_i=1.0515E+01 theta_v=-8.9570E+00 dA=8.43E-01 dB=1.19E+00 ERef=-1.700E-01.csv",
    f"ExpData Aug 7 rref={rref:.2E}\Exp Dimensionless sigma=9.7358E+02 theta_i=1.0515E+01 theta_v=-8.9570E+00 dA=8.43E-01 dB=1.19E+00 ERef=-1.700E-01.csv",
    f"ExpData Aug 7 rref={rref:.2E}\Exp Dimensionless sigma=1.9472E+03 theta_i=1.0515E+01 theta_v=-8.9570E+00 dA=8.43E-01 dB=1.19E+00 ERef=-1.700E-01.csv"
][:3]


# build a core network model
networkA = Network.build(name='NetworkA')


# build a PINN model
ECINN = PINN(networkA,name='ECINN').build(outerBoundary)

#weights folder is where weights is saved
if not os.path.exists('./weights'):
    os.mkdir('./weights')

default_weight_name = "./weights/default.h5"
ECINN.save_weights(default_weight_name)
#print(ECINN.trainable_variables) Inspect the names of trainable weights


def get_Lambda_values():
    lambda_theta_corr_value = [v for v in ECINN.trainable_variables if 'lambda_theta_corr' in v.name][0]
    lambda_dA_value = [v for v in ECINN.trainable_variables if 'lambda_d_A' in v.name][0]

    return lambda_theta_corr_value.numpy(),lambda_dA_value.numpy()

class AdaptiveWeightCallBack(tf.keras.callbacks.Callback):
    def __init__(self,weights_list):
        super().__init__()



        self.lambda_theta_corr_value_hist = []
        self.lambda_dA_value_hist = []
        self.lambda_dB_value_hist = []
        self.history_dict = dict()


    def on_epoch_end(self, epoch, logs=None):
        

        
        """
        if epoch>15 and epoch<19:
            self.BV_weight.assign_add(0.5)
        """ 


        lambda_theta_corr_value,lambda_dA_value = get_Lambda_values()
        print('theta corr',lambda_theta_corr_value,'dA',lambda_dA_value)

        self.lambda_theta_corr_value_hist.append(lambda_theta_corr_value)
        self.lambda_dA_value_hist.append(lambda_dA_value)


    def on_train_end(self,logs=None):


        self.history_dict['lambda_theta_corr'] = self.lambda_theta_corr_value_hist
        self.history_dict['lambda_d_A'] = self.lambda_dA_value_hist


        return self.history_dict



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,exp_file_name,sigma,theta_i,theta_v,PortionAnalyzed=1.0,Lambda=3.0,num_train_samples=1000000,batch_size=250):
        super().__init__()
        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.Lambda = Lambda
        self.PortionAnalyzed  = PortionAnalyzed
        self.theta_i = theta_i
        self.theta_v = theta_v


        self.exp_file_name = exp_file_name

        #Read Experimental Data
        self.df_exp = pd.read_csv(self.exp_file_name)

        self.sigma = sigma 

        self.FullScanT = 2.0*abs(theta_v-theta_i)/self.sigma
        self.maxT = self.FullScanT*PortionAnalyzed
        self.time_array = np.linspace(0,self.maxT,num=self.num_train_samples)
        np.random.shuffle(self.time_array)
        self.maxX = Lambda * np.sqrt(self.maxT)

    def __len__(self):
        return int(np.floor(self.num_train_samples / self.batch_size))


    def getFullScanT(self):
        return self.FullScanT
    
    def getMaxT(self):
        return self.maxT
    
    def getMaxX(self):
        return self.maxX
    
    def getDf_exp(self):
        return self.df_exp


    def __getitem__(self, index):
    
        # create training input
        #The diffusion equation domain

        TX_eqn_A = np.random.rand(self.batch_size, 2)
        TX_eqn_A[...,0] = TX_eqn_A[...,0] * self.maxT
        TX_eqn_A[..., 1] = TX_eqn_A[..., 1] * self.maxX




        # Initial condition T=0
        TX_ini_A = np.random.rand(self.batch_size, 2)  
        TX_ini_A[..., 0] = 0.0
        TX_ini_A[...,1] = TX_ini_A[...,1]*self.maxX      





        #Boundary condition at electrode surface (X=0)
        TX_bnd0_A = np.random.rand(self.batch_size, 2)         
        TX_bnd0_A[...,0] = np.sort(self.time_array[index*self.batch_size:(index+1)*self.batch_size])
        TX_bnd0_A[..., 1] =  0.0



        #Applied Potential
        theta_aux = np.random.rand(self.batch_size, 1)

        theta_aux[:,0] = np.where(TX_bnd0_A[...,0]<self.FullScanT/2.0,self.theta_i-self.sigma*TX_bnd0_A[...,0],self.theta_v+self.sigma*(TX_bnd0_A[...,0]-self.FullScanT/2.0))   


        interpolated_flux = exp_flux_sampling(TX_bnd0_A[...,0],self.df_exp,self.FullScanT,self.PortionAnalyzed)

        
        


        #Boundary condition at the outerboundary of simulatuion 
        TX_bnd1_A = np.random.rand(self.batch_size, 2)
        TX_bnd1_A[...,0] =  TX_bnd1_A[...,0] * self.maxT
        TX_bnd1_A[...,1] = self.maxX





        # create training output
        #Governing equation target
        C_eqn_A = np.zeros((self.batch_size, 1))


        #Initial equation target              
        C_ini_A = np.ones((self.batch_size,1))


        #Electrode Surface target    
        C_Nernst_bnd0_A = np.zeros((self.batch_size,1))



        #Outerboundary target
        if outerBoundary == 'SI':
            C_bnd1_A = np.ones((self.batch_size,1))

        elif outerBoundary =='TL': 
            C_bnd1_A = np.zeros((self.batch_size,1))

        else:
            raise ValueError

        x_train = [TX_eqn_A,TX_ini_A,TX_bnd0_A,TX_bnd1_A,theta_aux]
        y_train = [C_eqn_A,C_ini_A,C_Nernst_bnd0_A,interpolated_flux,C_bnd1_A]

        return x_train,y_train
    



def PINNFitExp(exp_file_name,scan_rate,sigma,theta_i,theta_v,dA_exp_measured,dB_exp_measured,epochs=200,PortionAnalyzed=1.0,Lambda=3.0,initial_weights=None,train=True,saving_directory='./Data'):
    """
    PortionAnalyzed=0.75: We only analyze the firt 75% of voltammogram, only cathodic reaction happens
    
    """
    # saving directory is where data(voltammogram, concentration profile etc is saved)
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)



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

    training_generator = DataGenerator(exp_file_name,sigma,theta_i,theta_v,PortionAnalyzed,Lambda,num_train_samples,batch_size)

    FullScanT =  training_generator.FullScanT
    maxT = FullScanT*PortionAnalyzed # Onlya certain part of voltammogram is anlyzed. 
    maxX = training_generator.maxX
    #Read Experimental Data
    df_exp = training_generator.df_exp







    file_name = f'sigma={sigma:.2E}'

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

    # The initial weights regarding to bnd0 (BV) boundary condition at electrode surface is set to 0 for a few epochs. Then slowly increased to 1 latter.
    weights_list = [1.0,1.0,1.0, 1.0,1.0]
    adptiveweightcallback = AdaptiveWeightCallBack(weights_list)

    ECINN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='mse',loss_weights = weights_list)
    ECINN.load_weights(default_weight_name)






    if train:
        if initial_weights:
            ECINN.load_weights(initial_weights)
        history =  ECINN.fit(training_generator,epochs=epochs,verbose=2,callbacks=[lr_scheduler_callback,adptiveweightcallback],workers=3,use_multiprocessing=False,max_queue_size=24)
        history_callback = adptiveweightcallback.history_dict
        df_history = pd.DataFrame({**history.history,**history_callback})
        df_history.to_csv(f'{saving_directory}/history {file_name}.csv')
        ECINN.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            ECINN.load_weights(f'./weights/weights {file_name}.h5')
            ECINN.save_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            if initial_weights:
                ECINN.load_weights(initial_weights)
            history =  ECINN.fit(training_generator,epochs=epochs,verbose=2,callbacks=[lr_scheduler_callback,adptiveweightcallback],workers=3,use_multiprocessing=False,max_queue_size=24)
            history_callback = adptiveweightcallback.history_dict
            df_history = pd.DataFrame({**history.history,**history_callback})

            df_history.to_csv(f'{saving_directory}/history {file_name}.csv')
            ECINN.save_weights(f'./weights/weights {file_name}.h5')

    #Get the trained variable for the inverse problem
    lambda_theta_corr_value,lambda_dA_value = get_Lambda_values()
    E0fCalc = ERef+lambda_theta_corr_value/38.9433


    # predict c(t,x) distribution
    t_flat = np.linspace(0, maxT, num_test_samples)
    cv_flat = np.where(t_flat<FullScanT/2.0,theta_i-sigma*t_flat,theta_v+sigma*(t_flat-FullScanT/2.0))
    x_flat = np.linspace(0, maxX, num_test_samples) 
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    cA = networkA.predict(tx, batch_size=num_test_samples)
    cA = cA.reshape(t.shape)



    TX_flux = np.zeros((len(t_flat),2))
    TX_flux[:,0] = t_flat

    TX_flux = tf.convert_to_tensor(TX_flux,dtype=tf.float32)

    with tf.GradientTape() as g:
        g.watch(TX_flux)
        C = networkA(TX_flux)

    dC_dX = g.batch_jacobian(C,TX_flux)[...,1]
    flux = -lambda_dA_value* dC_dX.numpy().reshape(-1)
    
    df = pd.DataFrame({'Potential':cv_flat,'Flux':flux})
    df.to_csv(f'{saving_directory}/PINN {file_name}.csv',index=False)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(cv_flat,flux,label='PINN prediction')
    #ax.axhline(-0.446*math.sqrt(sigma),label='R-S equation',ls='--',color='r')
    #ax.axvline(-1.109,label='Expected Forward Scan Potential',ls='--',color='k')
    ax.set_xlabel('Potential, theta')
    ax.set_ylabel('Flux,J')
    ax.legend()
    fig.savefig(f'{saving_directory}/CV {file_name}.png')



    # plot u(t,x) distribution as a color-map
    fig,axs = plt.subplots(figsize=(8,9),nrows=2)
    ax = axs[0]
    axes = ax.pcolormesh(t, x, cA, cmap='YlGnBu',shading='auto')

    ax.set_xlabel('T',fontsize='large',fontweight='bold')
    ax.set_ylabel('X',fontsize='large',fontweight='bold')
    #ax.set_ylim(0,maxX*0.4)
    cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)
    cbar.set_label(r'$C_A(T,X)$',fontsize='large',fontweight='bold')
    axes.set_clim(0.0, 1)





    ax = axs[1]
    df.plot(x='Potential',y='Flux',ax=ax,color='b',lw=3,alpha=0.7,label='PINN')


    df_exp.plot(x=0,y=1,ax=ax,color='r',ls='--',lw=3,alpha=0.7,label='Experiment')
    ax.axvline(x=lambda_theta_corr_value,lw=3,ls='--',color='r',label=r'$\theta_{corr}$')
    #ax.axhline(-0.446*math.sqrt(sigma),label='R-S Equation',ls='--',color='k',lw=3)
    #ax.plot([-5,5],[-0.446*math.sqrt(sigma),-0.446*math.sqrt(sigma)],label='R-S Equation',ls='--',color='k',lw=3)
    #ax.annotate("", xy=(2.5, -0.5), xytext=(10, -0.5),arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_xlabel(r'Potential,$\theta$')
    ax.set_ylabel(r'Flux, $J$')
    sec_ax = ax.secondary_xaxis(-0.15,functions=(toDimensionalPotential,toDimensionlessPotential))
    sec_ax.set_xlabel(f'$E-E_{{ref}}$, V')

    ax.legend(fontsize=15)

    fig.suptitle(f'$\\theta_{{corr}}$={lambda_theta_corr_value:.3f},$E_{{corr}}={lambda_theta_corr_value/38.9433:.4f}V$\nPredicted $E^0_f={E0fCalc:.4f}$ V vs. SCE\n$\lambda_{{d_A}}$ = {lambda_dA_value:.3f} $\lambda_{{D_A}}={lambda_dA_value*1e-9:.2E}m^2s^{{-1}}$')


    fig.text(0.05,0.92,'a)',fontsize=30)
    fig.text(0.05,0.5,'b)',fontsize=30)
    fig.tight_layout()
    fig.savefig(f'{saving_directory}/PINN {file_name}.png',dpi=250)

    plt.close('all')

    # Plot figure in dimensional form for the paper 
    fig,axs = plt.subplots(figsize=(8,9),nrows=2)
    ax = axs[0]

    #convert to dimensional
    t_dim = t*r_e*r_e/Dref
    x_dim = x*r_e * 1e3 # Change units to mm
    c_dim = cA*c_bulk

    axes = ax.pcolormesh(t_dim, x_dim, c_dim, cmap='YlOrRd',shading='auto')

    ax.set_xlabel('$t / s$',fontsize='large',fontweight='bold')
    ax.set_ylabel('$x / mm$',fontsize='large',fontweight='bold')
    #ax.set_ylim(0,maxX*0.4)
    cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)
    cbar.set_label(r'$C_A(T,X)$',fontsize='large',fontweight='bold')

    cbar.set_label(r'$c_{Ru(NH_3)^{3+}_6}(t,x)/mM$',fontsize='large',fontweight='bold')
    axes.set_clim(0.0, 1.0*c_bulk)

    ax = axs[1]

    df['Potential'] = df['Potential'] / (F/(R*T)) + ERef
    df['Flux'] = df['Flux'] * (F*A*c_bulk*Dref/r_e) * 1e6 # Change unit to micronA
    df.plot(x='Potential',y='Flux',ax=ax,color='b',lw=3,alpha=0.6,label=f'$\\nu={scan_rate:.3f}$V/s')

    df_exp.iloc[:,0] = df_exp.iloc[:,0]/ (F/(R*T)) + ERef
    df_exp.iloc[:,1] = df_exp.iloc[:,1] * (F*A*c_bulk*Dref/r_e)* 1e6 # Change unit to micronA
    ax.plot(df_exp.iloc[:,0],df_exp.iloc[:,1],color='b',ls='--',lw=3,alpha=0.7)

    ax.axvline(x=E0fCalc,lw=3,ls='--',color='r',label=r'$E^0_f$, ECINN-Nernst')
    
    ax.annotate("", xy=(0.6, -3), xytext=(0.75, -3),arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_ylabel(r'$I/\mu A$',fontsize="large",fontweight='bold')
    ax.set_xlabel(r'E/V vs. SCE',fontsize='large',fontweight='bold')
    ax.legend(fontsize=16)

    fig.text(0.01,0.99,'a)',fontsize=30)
    fig.text(0.01,0.5,'b)',fontsize=30)
    fig.tight_layout()
    fig.savefig(f'{saving_directory}/PINN Paper {file_name}.png',dpi=250,bbox_inches='tight')

    return f'./weights/weights {file_name}.h5'



if __name__ == "__main__":
    for epochs in [50]:
        for scan_rate,exp_dimensionless_file in zip(scan_rates,exp_dimensionless_files):
            sigma,theta_i,theta_v,dA,dB = expParameters(exp_dimensionless_file)
            df_exp = pd.read_csv(exp_dimensionless_file)

            saving_directory = f"data new epochs={epochs} rref={rref:.2E}"
            PINNFitExp(exp_dimensionless_file,scan_rate=scan_rate,sigma=sigma,theta_i=theta_i,theta_v=theta_v,dA_exp_measured=dA,dB_exp_measured=dB,epochs=epochs,train=False,saving_directory=saving_directory)
