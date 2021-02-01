import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DGMLayer(tf.keras.layers.Layer):
    """
    Description: Class for implementing a DGM layer
    """
    def __init__(self, S_l_layer, input_dim, num_nodes, activation, dtype=tf.float64):
        super(DGMLayer, self).__init__(dtype=dtype)
        self.S_l = S_l_layer
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.activation = activation

    def build(self, input_shape):
        self.U_z_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.W_z_l = self.add_weight(shape=(self.num_nodes, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_z_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)
        self.U_g_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.W_g_l = self.add_weight(shape=(self.num_nodes, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_g_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)
        self.U_r_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.W_r_l = self.add_weight(shape=(self.num_nodes, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_r_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)
        self.U_h_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.W_h_l = self.add_weight(shape=(self.num_nodes, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_h_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)

    def call(self, input):
        S_l = self.S_l(input)
        Z_l = self.activation(tf.matmul(input, self.U_z_l) + tf.matmul(S_l, self.W_z_l) + self.b_z_l)
        G_l = self.activation(tf.matmul(input, self.U_g_l) + tf.matmul(S_l, self.W_g_l) + self.b_g_l)
        R_l = self.activation(tf.matmul(input, self.U_r_l) + tf.matmul(S_l, self.W_r_l) + self.b_r_l)
        H_l = self.activation(tf.matmul(input, self.U_h_l) + tf.matmul(tf.multiply(S_l, R_l), self.W_h_l) + self.b_h_l)
        return tf.multiply(tf.ones_like(G_l) - G_l, H_l) + tf.multiply(Z_l, S_l)



class DGMModel(tf.keras.models.Model):
    """
    Description: Class for implementing the DGM architechture
    """
    def __init__(self, input_dim=1, num_nodes=50, num_dgm_layers=3, activation=tf.keras.activations.tanh, dtype=tf.float64):
        super(DGMModel, self).__init__(dtype=dtype)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.num_dgm_layers = num_dgm_layers
        self.activation = activation
        self.S_layers = [tf.keras.layers.Dense(units=self.num_nodes, activation=self.activation, name='S_1', dtype=dtype)]
        for i in range(num_dgm_layers):
            self.S_layers.append(DGMLayer(S_l_layer=self.S_layers[i], input_dim=self.input_dim, num_nodes=self.num_nodes, activation=self.activation, dtype=dtype))
        self.f_layer = tf.keras.layers.Dense(units=1, use_bias=True, activation=None, name='f_layer', dtype=dtype)

    def call(self, input):
         x = self.S_layers[self.num_dgm_layers](input)
         return self.f_layer(x)

    def plot(self, interval = [0.0, 2*np.pi]):
        x = np.linspace(interval[0], interval[1], 100)
        x_t = tf.constant(np.reshape(x, (100, 1)), dtype = tf.float32)
        y_t = self.call(x_t)
        plt.plot(x_t.numpy(), y_t.numpy())
        plt.plot(x, np.sin(x))
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()

class DiffLayer(tf.keras.layers.Layer):
    def __init__(self, func):
        super(DiffLayer, self).__init__()
        self.func = func

    def call(self, input):
        with tf.GradientTape() as tape:
            tape.watch(input)
            return tape.gradient(self.func(input), input)


x = tf.constant([[1], [3]], dtype = tf.float64)
smodel = DGMModel(dtype = tf.float32)
print(smodel(x))
print(smodel.summary())
dsmodel = DiffLayer(smodel)
d2smodel = DiffLayer(smodel)


def loss0(input):
    zero = tf.constant([[0.]], dtype=tf.float32)
    loss_0 = tf.reduce_mean(tf.square(d2smodel(input) + smodel(input)))
    loss_1 = tf.square(dsmodel(zero) - 1.0)[0][0]
    loss_2 = tf.square(smodel(zero))[0][0]
    return loss_0 + loss_1 + loss_2

def loss1(input):
    zero = tf.constant([[0.]], dtype=tf.float32)
    loss_0 = tf.reduce_mean(tf.square(dsmodel(input) - tf.cos(input)))
    loss_1 = tf.square(smodel(zero))[0][0]
    return loss_0 + loss_1

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0001, decay_steps=1000, decay_rate=0.96, staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
for i in range(1000):
    x = tf.convert_to_tensor(np.random.rand(100, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss_ = loss1(x)
        grads = tape.gradient(loss_, smodel.trainable_weights)
        optimizer.apply_gradients(zip(grads, smodel.trainable_weights))
    print(loss_)
    if loss_ < 1e-16:
        break

print(smodel.trainable_variables)
smodel.plot()
