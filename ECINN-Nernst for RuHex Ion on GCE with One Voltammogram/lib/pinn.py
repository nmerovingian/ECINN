import tensorflow as tf
from .layer import GradientLayer,NernstLayer,DiffusionCoefficientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for Fick's law equation.
    """

    def __init__(self, networkA,name):


        self.networkA = networkA

        self.gradsA = GradientLayer(self.networkA)

        self.NernstLayer = NernstLayer()
        self.DLayer = DiffusionCoefficientLayer()
        self.name = name 
 

    def build(self,outerBoundary):

        tx_eqn_A = tf.keras.layers.Input(shape=(2,))

        tx_ini_A = tf.keras.layers.Input(shape=(2,))

        tx_bnd0_A = tf.keras.layers.Input(shape=(2,))

        tx_bnd1_A = tf.keras.layers.Input(shape=(2,))





        theta_aux = tf.keras.layers.Input(shape=(1,))


        # compute domain gradients
        u_A, du_dt_A, du_dx_A, d2u_dx2_A = self.gradsA(tx_eqn_A)



        # initial condition output
        u_ini_A = self.networkA(tx_ini_A)
        u_ini_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Ini_A')(u_ini_A)



        u_bnd0_A,du_dt_bnd0_A, du_dx_bnd0_A, d2u_dx2_bnd0_A = self.gradsA(tx_bnd0_A)






        u_eqn_A,flux_A = self.DLayer(du_dx_bnd0_A,du_dt_A,d2u_dx2_A)
        
        predicted_flux_A = -flux_A




        u_eqn_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Domain_A')(u_eqn_A)


        u_Nernst_bnd0_A = self.NernstLayer(theta_aux,u_bnd0_A)
        u_Nernst_bnd0_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Nernst_bnd0_A')(u_Nernst_bnd0_A)



        # boundary condition output
        if outerBoundary == 'SI':
            u_bnd1_A = self.networkA(tx_bnd1_A)

        elif outerBoundary == 'TL':
            ubnd1_A,du_dt_bnd1_A, du_dx_bnd1_A, d2u_dx2_bnd1_A = self.gradsA(tx_bnd1_A)
            u_bnd1_A = du_dx_bnd1_A


        u_bnd1_A = tf.keras.layers.Dense(1,activation=None,use_bias=False,kernel_initializer='ones',trainable=False,name='Outer_A')(u_bnd1_A)


        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[tx_eqn_A,tx_ini_A,tx_bnd0_A, tx_bnd1_A, theta_aux], outputs=[u_eqn_A,u_ini_A,u_Nernst_bnd0_A,predicted_flux_A,u_bnd1_A],name = self.name)
