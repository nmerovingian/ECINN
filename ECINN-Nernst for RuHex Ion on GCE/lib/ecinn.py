import tensorflow as tf
from .layer import GradientLayer,NernstLayer,DiffusionCoefficientLayer

class ECINN:
    """
    Build a physics informed neural network (PINN) model for Fick's law equation.
    """

    def __init__(self, network_1_A,network_2_A,network_3_A,name):


        self.network_1_A = network_1_A


        self.network_2_A = network_2_A


        self.network_3_A = network_3_A



        self.grads_1_A = GradientLayer(self.network_1_A)


        self.grads_2_A = GradientLayer(self.network_2_A)


        self.grads_3_A = GradientLayer(self.network_3_A)



        self.NernstLayer = NernstLayer()
        self.DLayer = DiffusionCoefficientLayer()
        self.name = name 
 

    def build(self,outerBoundary):

        tx_eqn_1_A = tf.keras.layers.Input(shape=(2,))
        tx_ini_1_A = tf.keras.layers.Input(shape=(2,))
        tx_bnd0_1_A = tf.keras.layers.Input(shape=(2,))
        tx_bnd1_1_A = tf.keras.layers.Input(shape=(2,))


        theta_aux_1 = tf.keras.layers.Input(shape=(1,))


        tx_eqn_2_A = tf.keras.layers.Input(shape=(2,))
        tx_ini_2_A = tf.keras.layers.Input(shape=(2,))
        tx_bnd0_2_A = tf.keras.layers.Input(shape=(2,))
        tx_bnd1_2_A = tf.keras.layers.Input(shape=(2,))


        theta_aux_2 = tf.keras.layers.Input(shape=(1,))


        tx_eqn_3_A = tf.keras.layers.Input(shape=(2,))
        tx_ini_3_A = tf.keras.layers.Input(shape=(2,))
        tx_bnd0_3_A = tf.keras.layers.Input(shape=(2,))
        tx_bnd1_3_A = tf.keras.layers.Input(shape=(2,))

        theta_aux_3 = tf.keras.layers.Input(shape=(1,))




        # compute domain gradients
        u_1_A, du_dt_1_A, du_dx_1_A, d2u_dx2_1_A = self.grads_1_A(tx_eqn_1_A)

        u_2_A, du_dt_2_A, du_dx_2_A, d2u_dx2_2_A = self.grads_2_A(tx_eqn_2_A)

        u_3_A, du_dt_3_A, du_dx_3_A, d2u_dx2_3_A = self.grads_3_A(tx_eqn_3_A)



        # initial condition output
        u_ini_1_A = self.network_1_A(tx_ini_1_A)
        u_ini_1_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Ini_1_A')(u_ini_1_A)


        u_ini_2_A = self.network_2_A(tx_ini_2_A)
        u_ini_2_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Ini_2_A')(u_ini_2_A)


        u_ini_3_A = self.network_3_A(tx_ini_3_A)
        u_ini_3_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Ini_3_A')(u_ini_3_A)




        u_bnd0_1_A,du_dt_bnd0_1_A, du_dx_bnd0_1_A, d2u_dx2_bnd0_1_A = self.grads_1_A(tx_bnd0_1_A)
        u_bnd0_2_A,du_dt_bnd0_2_A, du_dx_bnd0_2_A, d2u_dx2_bnd0_2_A = self.grads_2_A(tx_bnd0_2_A)
        u_bnd0_3_A,du_dt_bnd0_3_A, du_dx_bnd0_3_A, d2u_dx2_bnd0_3_A = self.grads_3_A(tx_bnd0_3_A)




        u_eqn_1_A,flux_1_A, = self.DLayer(du_dx_bnd0_1_A,du_dt_1_A,d2u_dx2_1_A)
        
        predicted_flux_1_A = -flux_1_A


        u_eqn_2_A,flux_2_A = self.DLayer(du_dx_bnd0_2_A,du_dt_2_A,d2u_dx2_2_A)
        
        predicted_flux_2_A = -flux_2_A



        u_eqn_3_A,flux_3_A, = self.DLayer(du_dx_bnd0_3_A,du_dt_3_A,d2u_dx2_3_A)
        
        predicted_flux_3_A = -flux_3_A



        u_eqn_1_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Domain_1_A')(u_eqn_1_A)

        u_eqn_2_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Domain_2_A')(u_eqn_2_A)

        u_eqn_3_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Domain_3_A')(u_eqn_3_A)


        u_Nernst_bnd0_1_A = self.NernstLayer(theta_aux_1,u_bnd0_1_A)
        u_Nernst_bnd0_1_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Nernst_bnd0_1_A')(u_Nernst_bnd0_1_A)

        u_Nernst_bnd0_2_A = self.NernstLayer(theta_aux_2,u_bnd0_2_A)
        u_Nernst_bnd0_2_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Nernst_bnd0_2_A')(u_Nernst_bnd0_2_A)

        u_Nernst_bnd0_3_A= self.NernstLayer(theta_aux_3,u_bnd0_3_A)
        u_Nernst_bnd0_3_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Nernst_bnd0_3_A')(u_Nernst_bnd0_3_A)





        # boundary condition output
        if outerBoundary == 'SI':
            u_bnd1_1_A = self.network_1_A(tx_bnd1_1_A)
            u_bnd1_2_A = self.network_2_A(tx_bnd1_2_A)
            u_bnd1_3_A = self.network_2_A(tx_bnd1_3_A)
                      
        elif outerBoundary == 'TL':
            ubnd1_1_A,du_dt_bnd1_1_A, du_dx_bnd1_1_A, d2u_dx2_bnd1_1_A = self.grads_1_A(tx_bnd1_1_A)
            u_bnd1_1_A = du_dx_bnd1_1_A


            ubnd1_2_A,du_dt_bnd1_2_A, du_dx_bnd1_2_A, d2u_dx2_bnd1_2_A = self.grads_2_A(tx_bnd1_2_A)
            u_bnd1_2_A = du_dx_bnd1_2_A

            ubnd1_3_A,du_dt_bnd1_3_A, du_dx_bnd1_3_A, d2u_dx2_bnd1_3_A = self.grads_3_A(tx_bnd1_3_A)
            u_bnd1_3_A = du_dx_bnd1_3_A


        u_bnd1_1_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='bnd1_1_A')(u_bnd1_1_A)

        u_bnd1_2_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='bnd1_2_A')(u_bnd1_2_A)

        u_bnd1_3_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='bnd1_3_A')(u_bnd1_3_A)



        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[tx_eqn_1_A, tx_ini_1_A, tx_bnd0_1_A,tx_bnd1_1_A,theta_aux_1] + [tx_eqn_2_A, tx_ini_2_A, tx_bnd0_2_A,tx_bnd1_2_A,theta_aux_2] + [tx_eqn_3_A, tx_ini_3_A, tx_bnd0_3_A,tx_bnd1_3_A,theta_aux_3],
            outputs=[u_eqn_1_A,u_ini_1_A,u_Nernst_bnd0_1_A,predicted_flux_1_A,u_bnd1_1_A]+[u_eqn_2_A,u_ini_2_A,u_Nernst_bnd0_2_A,predicted_flux_2_A,u_bnd1_2_A]+[u_eqn_3_A,u_ini_3_A,u_Nernst_bnd0_3_A,predicted_flux_3_A,u_bnd1_3_A],
            name = self.name)
