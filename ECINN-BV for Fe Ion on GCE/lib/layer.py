import tensorflow as tf



class min_max_clipping(tf.keras.constraints.Constraint):
    def __init__(self,min,max) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def __call__(self, w):
        return tf.cast(tf.clip_by_value(w,clip_value_min=self.min,clip_value_max=self.max),w.dtype)
        



class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for Fick's law of diffusion

    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, x):
        """
        Computing 1st and 2nd derivatives for Fick's equation.

        Args:
            x: input variable.

        Returns:
            model output, 1st and 2nd derivatives.
        """

        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                u = self.model(x)
            dc_dtx = gg.batch_jacobian(u, x)
            dc_dt = dc_dtx[..., 0]
            dc_dx = dc_dtx[..., 1]
        d2c_dx2 = g.batch_jacobian(dc_dx, x)[..., 1]
        return u, dc_dt, dc_dx, d2c_dx2
    

class BVLayer(tf.keras.layers.Layer):
    """
    Custom layer for BV kinetics for full reaction

    """
    def __init__(self,lambda_k0=5.0,lambda_alpha=0.4,lambda_beta=0.4):
        super().__init__()
        self.lambda_K0 = tf.Variable(initial_value=lambda_k0,trainable=True,name='lambda_K0_G')
        self.lambda_alpha = tf.Variable(initial_value=lambda_alpha,trainable=True,name='lambda_alpha_G')
        self.lambda_beta = tf.Variable(initial_value=lambda_beta,trainable=True,name='lambda_beta_G')
        
        
    def call(self,theta_aux,u_bnd0):
        return self.lambda_K0*tf.exp(-self.lambda_alpha*theta_aux)*u_bnd0 - self.lambda_K0*tf.exp(self.lambda_beta*theta_aux)*(1.0-u_bnd0)

class BVLayerCathodic(tf.keras.layers.Layer):
    """
    Custom layer for BV kinetics for the cathodic scan. Only reduction matters.
    """
    def __init__(self,name='FluxLayer'):
        super().__init__(name=name)
        self.lambda_K0 = tf.Variable(initial_value=1.0,trainable=True,name='lambda_K0',constraint=tf.keras.constraints.non_neg())
        self.lambda_alpha = tf.Variable(initial_value=0.4,trainable=True,name='lambda_alpha',constraint=tf.keras.constraints.non_neg())

    def call(self,theta_aux,u_bnd0):
        return self.lambda_K0*tf.exp(-self.lambda_alpha*theta_aux)*u_bnd0
    


class DiffusionCoefficientLayer(tf.keras.layers.Layer):
    
    def __init__(self):
        super().__init__()
        self.lambda_dA = tf.Variable(initial_value=0.4,trainable=True,name ='lambda_d_A',constraint=tf.keras.constraints.non_neg())

    def call(self,du_dx_bnd0,du_dt,d2u_dx2):
        flux = self.lambda_dA * du_dx_bnd0

        u_eqn  = du_dt-self.lambda_dA*d2u_dx2


        return  flux,u_eqn
    


class BayesianDiffusionCoefficientLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.lambda_dA_mu = tf.Variable(initial_value=0.4,trainable=True,name ='lambda_d_A_mu',constraint=tf.keras.constraints.min_max_norm(min_value=0.5,max_value=0.3,rate=0.9,axis=None))
        self.lambda_dA_logvar = tf.Variable(initial_value=0.0,trainable=True,name ='lambda_d_A_logvar')

    
    def call(self,du_dx_bnd0,du_dt,d2u_dx2):
        epsilon = tf.keras.backend.random_normal(shape=self.lambda_dA_mu.shape)
        sampled_lambda_dA = self.lambda_dA_mu +tf.keras.backend.exp(self.lambda_dA_logvar/2) * epsilon

        flux = sampled_lambda_dA * du_dx_bnd0
        u_eqn  = du_dt-sampled_lambda_dA*d2u_dx2

        kl_loss = -0.5 * tf.keras.backend.sum(1+self.lambda_dA_logvar-tf.keras.backend.square(self.lambda_dA_mu) - tf.keras.backend.exp(self.lambda_dA_logvar),axis=None)

        return flux,u_eqn,kl_loss

