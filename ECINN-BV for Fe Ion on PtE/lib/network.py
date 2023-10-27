import tensorflow as tf

class Network:
    """
    Build a physics informed neural network (PINN) model for Fick's equation.
    """

    @classmethod
    def build(cls, num_inputs=2, layers=[64,32, 32, 32,64], activation='tanh', num_outputs=1,name=None):
        

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                kernel_initializer='he_normal')(x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='he_normal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs,name=name)


