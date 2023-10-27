import tensorflow as tf
from .layer import GradientLayer,BVLayerCathodic,DiffusionCoefficientLayer

class ECINN:
    """
    Build a physics informed neural network (PINN) model for Fick's law equation.
    """

    def __init__(self, network1,network2,network3):


        self.network1 = network1
        self.network2 = network2 
        self.network3 = network3
        self.grads1 = GradientLayer(self.network1)
        self.grads2 = GradientLayer(self.network2)
        self.grads3 = GradientLayer(self.network3)
        self.BVLayer = BVLayerCathodic()
        self.DLayer = DiffusionCoefficientLayer()

 

    def build(self,outerBoundary):
        # Input for the voltammogram at a low scan rate
        tx_eqn_1 = tf.keras.layers.Input(shape=(2,),name='Domain_Input_1')
        tx_ini_1 = tf.keras.layers.Input(shape=(2,),name='Ini_Input_1')
        tx_bnd0_1 = tf.keras.layers.Input(shape=(2,),name='Bnd0_Input_1')
        tx_bnd1_1 = tf.keras.layers.Input(shape=(2,),name='Bnd1_Input_1')
        theta_aux_1 = tf.keras.layers.Input(shape=(1,),name="Potential_Input_1")

        
        # Input for the voltammogram at a low scan rate
        tx_eqn_2 = tf.keras.layers.Input(shape=(2,),name='Domain_Input_2')
        tx_ini_2 = tf.keras.layers.Input(shape=(2,),name='Ini_Input_2')
        tx_bnd0_2 = tf.keras.layers.Input(shape=(2,),name='Bnd0_Input_2')
        tx_bnd1_2 = tf.keras.layers.Input(shape=(2,),name='Bnd1_Input_2')
        theta_aux_2 = tf.keras.layers.Input(shape=(1,),name="Potential_Input_2")


        # Input for the voltammogram at a medium scan rate
        tx_eqn_3 = tf.keras.layers.Input(shape=(2,),name='Domain_Input_3')
        tx_ini_3 = tf.keras.layers.Input(shape=(2,),name='Ini_Input_3')
        tx_bnd0_3 = tf.keras.layers.Input(shape=(2,),name='Bnd0_Input_3')
        tx_bnd1_3 = tf.keras.layers.Input(shape=(2,),name='Bnd1_Input_3')
        theta_aux_3 = tf.keras.layers.Input(shape=(1,),name="Potential_Input_3")


        # compute gradients
        u_1, du_dt_1, du_dx_1, d2u_dx2_1 = self.grads1(tx_eqn_1)
        u_2, du_dt_2, du_dx_2, d2u_dx2_2 = self.grads2(tx_eqn_2)
        u_3, du_dt_3, du_dx_3, d2u_dx2_3 = self.grads3(tx_eqn_3)

        # initial condition output
        u_ini_1 = self.network1(tx_ini_1)
        u_ini_2 = self.network2(tx_ini_2)
        u_ini_3 = self.network3(tx_ini_3)



        #Compute electrode surface gradients
        u_bnd0_1,du_dt_bnd0_1, du_dx_bnd0_1, d2u_dx2_bnd0_1 = self.grads1(tx_bnd0_1)
        u_bnd0_2,du_dt_bnd0_2, du_dx_bnd0_2, d2u_dx2_bnd0_2 = self.grads2(tx_bnd0_2)
        u_bnd0_3,du_dt_bnd0_3, du_dx_bnd0_3, d2u_dx2_bnd0_3 = self.grads3(tx_bnd0_3)



        diffusion_flux_1,u_eqn_1 = self.DLayer(du_dx_bnd0_1,du_dt_1,d2u_dx2_1)
        diffusion_flux_2,u_eqn_2 = self.DLayer(du_dx_bnd0_2,du_dt_2,d2u_dx2_2)
        diffusion_flux_3,u_eqn_3 = self.DLayer(du_dx_bnd0_3,du_dt_3,d2u_dx2_3)



        predicted_flux_1 = -diffusion_flux_1
        predicted_flux_2 = -diffusion_flux_2
        predicted_flux_3 = -diffusion_flux_3

        predicted_flux_1 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='predicted_flux_1')(predicted_flux_1)
        predicted_flux_2 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='predicted_flux_2')(predicted_flux_2)
        predicted_flux_3 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='predicted_flux_3')(predicted_flux_3)




        BV_flux_1 = self.BVLayer(theta_aux_1,u_bnd0_1)
        BV_flux_2 = self.BVLayer(theta_aux_2,u_bnd0_2)
        BV_flux_3 = self.BVLayer(theta_aux_3,u_bnd0_3)






        u_BV_bnd0_1 =  diffusion_flux_1 - BV_flux_1
        u_BV_bnd0_2 =  diffusion_flux_2 - BV_flux_2
        u_BV_bnd0_3 =  diffusion_flux_3 - BV_flux_3

        u_BV_bnd0_1 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='BV_bnd0_1')(u_BV_bnd0_1)
        u_BV_bnd0_2 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='BV_bnd0_2')(u_BV_bnd0_2)
        u_BV_bnd0_3 = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='BV_bnd0_3')(u_BV_bnd0_3)


        # boundary condition output
        if outerBoundary == 'SI':
            u_bnd1_1 = self.network1(tx_bnd1_1)
            u_bnd1_2 = self.network2(tx_bnd1_2)
            u_bnd1_3 = self.network3(tx_bnd1_3)

        elif outerBoundary == "TL":
            u_bnd1_1,du_dt_bnd1_1, du_dx_bnd1_1, d2u_dx2_bnd1_1 = self.grads1(tx_bnd1_1)
            u_bnd1_2,du_dt_bnd1_2, du_dx_bnd1_2, d2u_dx2_bnd1_2 = self.grads2(tx_bnd1_2)
            u_bnd1_3,du_dt_bnd1_3, du_dx_bnd1_3, d2u_dx2_bnd1_3 = self.grads3(tx_bnd1_3)








        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[tx_eqn_1, tx_ini_1, tx_bnd0_1,tx_bnd1_1,theta_aux_1,tx_eqn_2, tx_ini_2, tx_bnd0_2,tx_bnd1_2,theta_aux_2,tx_eqn_3, tx_ini_3, tx_bnd0_3,tx_bnd1_3,theta_aux_3], 
            outputs=[u_eqn_1,u_eqn_2,u_eqn_3, u_ini_1,u_ini_2,u_ini_3,u_BV_bnd0_1,u_BV_bnd0_2,u_BV_bnd0_3,predicted_flux_1,predicted_flux_2,predicted_flux_3,u_bnd1_1,u_bnd1_2,u_bnd1_3])
