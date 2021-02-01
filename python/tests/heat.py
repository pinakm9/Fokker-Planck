# add required folders to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
image_dir = str(script_dir.parent.parent)
print(image_dir)
sys.path.insert(0, module_dir + '/modules')
# import modules
import tensorflow as tf
import numpy as np
import nnplot as nnp
import utility as ut
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')
class DGMLayer(tf.keras.layers.Layer):
    """
    Description: Class for implementing a DGM layer
    """
    def __init__(self, S_l_layer, input_dim, num_nodes, activation, dtype=tf.float32):
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
    def __init__(self, input_dim=1, num_nodes=50, num_dgm_layers=3, activation=tf.keras.activations.tanh, dtype=tf.float32):
        super(DGMModel, self).__init__(dtype=dtype)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.num_dgm_layers = num_dgm_layers
        self.activation = activation
        self.S_layers = [tf.keras.layers.Dense(units=self.num_nodes, activation=self.activation, name='S_1', dtype=dtype)]
        for i in range(num_dgm_layers):
            self.S_layers.append(DGMLayer(S_l_layer=self.S_layers[i], input_dim=self.input_dim, num_nodes=self.num_nodes, activation=self.activation, dtype=dtype))
        self.f_layer = tf.keras.layers.Dense(units=1, use_bias=True, activation=None, name='f_layer', dtype=dtype)

    def call(self, *args):
        x = tf.concat(args, 1)
        x = self.S_layers[self.num_dgm_layers](x)
        return self.f_layer(x)

class Dx(tf.keras.layers.Layer):
    def __init__(self, func):
        super(Dx, self).__init__()
        self.func = func

    @tf.function
    def call(self, t, x):
        z = self.func(t, x)
        return tf.gradients(z, x)[0]

class Dt(tf.keras.layers.Layer):
    def __init__(self, func):
        super(Dt, self).__init__()
        self.func = func

    @tf.function
    def call(self, t, x):
        z = self.func(t, x)
        return tf.gradients(z, t)[0]


x = tf.constant([[1], [3]], dtype = tf.float32)
t = tf.constant([[2], [4]], dtype = tf.float32)
h = DGMModel(2, dtype=tf.float32)
h_x = Dx(h)
h_xx = Dx(h_x)
h_t = Dt(h)
L = np.pi

def loss(x0, t0, x1, t1):
    loss_0 = tf.reduce_mean(tf.square(h_t(t0, x0) - h_xx(t0, x0)))
    loss_1 = tf.reduce_mean(tf.square(h(tf.zeros_like(x1), x1) - tf.sin(x1)))
    loss_2 = tf.reduce_mean(tf.square(h(t1, tf.zeros_like(t1))))
    loss_3 = tf.reduce_mean(tf.square(h(t1, L*tf.ones_like(t1))))
    return loss_0 + loss_1 + loss_2 + loss_3

def diff_op(t, x):
    with tf.GradientTape() as outer_x:
        outer_x.watch(x)
        with tf.GradientTape() as inner:
            inner.watch([t, x])
            p_ = h(t, x)
            grad_p = inner.gradient(p_, [t, x])
        p_t = grad_p[0]
        p_x = grad_p[1]
    p_xx = outer_x.gradient(p_x, x)
    return tf.reduce_mean( (-p_t + p_xx)**2 )

print('diff_op comparison {} {}'.format(diff_op(t, x), tf.reduce_mean(tf.square(h_t(t, x) - h_xx(t, x)))))
def true_sol(t, x):
    return np.exp(-t)*np.sin(x)

def plot_true_vs_nn(t = 0.5):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111)
    x = np.linspace(0., L, 100)
    x_t = tf.constant(np.reshape(x, (100, 1)), dtype = tf.float32)
    ax.plot(x, true_sol(t, x))
    ax.plot(x, h(t*tf.ones_like(x_t), x_t))
    plt.savefig('../../images/heat_sols.png')


def plot_true(t = 0.5):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111)
    x = np.linspace(0., L, 100)
    x_t = tf.constant(np.reshape(x, (100, 1)), dtype = tf.float32)
    ax.plot(x, true_sol(t, x))
    plt.savefig('../../images/heat_true.png')

def plot_nn(t = 0.5):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111)
    x = np.linspace(0., L, 100)
    x_t = tf.constant(np.reshape(x, (100, 1)), dtype = tf.float32)
    ax.plot(x, h(t*tf.ones_like(x_t), x_t))
    plt.savefig('../../images/heat_nn.png')

def error_heatmap(s=50):
    err = np.zeros((s, s))
    for i in range(s):
        x = (L*i)/(s-1)
        x_t = tf.constant([[x]], dtype=tf.float32)
        for j in range(s):
            t = float(j)/(s-1)
            t_t = tf.constant([[t]], dtype=tf.float32)
            true = true_sol(t, x)
            nn = h(t_t, x_t).numpy()[0]
            err[i][j] = abs(true - nn)

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(err, cmap='viridis')
    fig.colorbar(im)
    plt.savefig('../../images/heat_error.png')



#print(h(x, t))
#print(h_x(x, t))

# Time limits
T0 = 0.0 + 1e-10    # Initial time
T  = 1.0            # Terminal time

# Space limits
S1 = 0.0 + 1e-10    # Low boundary
S2 = L


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=1000, decay_rate=0.96, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
steps_per_sample = 1
n1 = 1000
n2 = 100
for i in range(200):
    x0 = tf.convert_to_tensor(np.random.uniform(low=S1, high=S2, size = (n1, 1)), dtype=tf.float32)
    t0 = tf.convert_to_tensor(np.random.uniform(low=T0, high=T, size = (n1, 1)), dtype=tf.float32)
    x1 = tf.convert_to_tensor(np.random.uniform(low=S1, high=S2, size = (n2, 1)), dtype=tf.float32)
    t1 = tf.convert_to_tensor(np.random.uniform(low=T0, high=T, size = (n2, 1)), dtype=tf.float32)
    for j in range(steps_per_sample):
        with tf.GradientTape() as tape:
            loss_ = loss(x0, t0, x1, t1)
            grads = tape.gradient(loss_, h.trainable_weights)
            optimizer.apply_gradients(zip(grads, h.trainable_weights))
        print(loss_)
    if loss_ < 1e-16:
        break

plot_true(t=0.5)
plot_nn(t=0.5)
plot_true_vs_nn(t=0.5)
error_heatmap()

alpha = 1.0
class Sol(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=h.dtype)
    def call(self, t, x):
        return tf.exp(-t)*tf.sin(x)

s = Sol()
plotter = nnp.NNPlotter(funcs=[s, h], space=[[0., L]], num_pts_per_dim=40)
plotter.plot(file_path='../../images/{}.png'.format('heat_eqn'), style='standard', t=0.5, x_lim=[0., L], y_lim=None, z_lim=None)
plotter.animate(file_path='../../images/{}.mp4'.format('heat_eqn'), style='standard', t=[T0, T], x_lim=[0., L], y_lim=None, z_lim=None)
