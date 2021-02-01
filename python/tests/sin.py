import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def call(self, input):
        return input

class DenseLayer(tf.keras.layers.Layer):
    """
    Description: Class for implementing a Dense layer
    """
    def __init__(self, S_l_layer, input_dim, num_nodes, activation=tf.keras.activations.tanh):
        super(DenseLayer, self).__init__()
        self.S_l = S_l_layer
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.activation = activation

    def build(self, input_shape):
        self.U_z_l = self.add_weight(shape=(self.input_dim, self.num_nodes), initializer="random_normal", trainable=True)
        self.b_z_l = self.add_weight(shape=(self.num_nodes, ), initializer="random_normal", trainable=True)

    def call(self, input):
        S_l = self.S_l(input)
        Z_l = self.activation(tf.matmul(input, self.U_z_l) + self.b_z_l)
        return Z_l

def diff_op(func):
    def loss(input):
        with tf.GradientTape() as tape:
            tape.watch(input)
            y = func(input)
            dy_dx = tape.gradient(y, input)
            return tf.math.pow(dy_dx - tf.math.cos(input), 2) + tf.math.pow(func(tf.zeros_like(input)), 2)
    return loss



class QuadModel(tf.keras.models.Model):
    def __init__(self, input_dim=1,  num_nodes=500, activation=tf.keras.activations.tanh):
        super(QuadModel, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.activation =  activation
        self.dl_0 = IdentityLayer()
        self.dl_1 = DenseLayer(self.dl_0, self.input_dim, self.num_nodes)
        self.dl_2 = DenseLayer(self.dl_1, self.input_dim, self.num_nodes)
        self.dl_3 = DenseLayer(self.dl_2, self.input_dim, self.num_nodes)
        self.dl_4 = DenseLayer(self.dl_3, self.input_dim, self.num_nodes)
        self.dl_5 = DenseLayer(self.dl_4, self.input_dim, 1)
        self.objective = diff_op(self.dl_5)

    def call(self, input):
        y = self.dl_5(input)
        objective = self.objective(input)
        print('loss = {}'.format(objective))
        self.add_loss(objective)
        return y

    def plot(self, interval = [0.0, 1]):
        x = np.linspace(interval[0], interval[1], 100)
        x_t = tf.constant(np.reshape(x, (100, 1)), dtype = tf.float32)
        y_t = self.dl_5(x_t)
        plt.plot(x_t.numpy(), y_t.numpy())
        plt.plot(x, np.sin(x))
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()



x = tf.constant([[1], [3]], dtype = tf.float64)
qmodel = QuadModel()
qmodel.compile(optimizer = 'sgd')
print(qmodel(x))
print(qmodel.summary())

data_x = tf.constant(np.random.rand(1000000, 1))
data_y = tf.constant(np.zeros((1000000, 1)))
qmodel.fit(x = data_x, y = data_y, epochs = 1)
qmodel.plot()
