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
import integrate as quad
import utility as ut
import dgm

# exponential ansatz
D = 0.5
R = 5.0
beta = 1.0
gamma = 1.0
space_r = np.array([0.0, R])
space_theta = np.array([-np.pi, np.pi])
space_x = gamma * R * np.array([-1., 1.])
space_y = gamma * R * np.array([-1., 1.])
time_ = np.array([0.0, 10.0])
num_nodes = 50
num_layers = 3
nn_type = 'LSTM'
model_name = 'cricle8_{}_{}_{}'.format(num_nodes, num_layers, str(D).replace('.', '_'))


# define f
f = dgm.DGM(dim = 2, num_nodes=num_nodes, num_layers = num_layers, final_activation=tf.square, dtype=tf.float32,\
            regularizer = tf.keras.regularizers.L2(l2=0.01))
f.add_domain(time_, 'uniform', space_r, 'uniform')
log_2_pi = tf.cast(tf.math.log(2.0 * np.pi), f.dtype)
sqrt2 = tf.cast(tf.sqrt(2.0), f.dtype)
#"""
try:
    f.load_weights('saved models/' + model_name).expect_partial()
except:
    pass
#"""
print(f(*f.domains[0].sample(5)))
f.summary()


# define c
c = dgm.DGM(dim = 1, num_nodes=num_nodes, num_layers = num_layers, final_activation=tf.square, dtype=f.dtype,\
            regularizer = tf.keras.regularizers.L2(l2=0.01))
c.add_domain(time_, 'uniform')
#"""
try:
    c.load_weights('saved models/' + model_name + '_norm').expect_partial()
except:
    pass
#"""

# define initial condition
init_cond = lambda r: tf.reduce_mean((f(tf.zeros_like(r), r) - 0.5*r*r - log_2_pi)**2)

# define differential operator
@ut.timer
def diff_op(t, r):
    r2 = r*r
    z = 4.0*(r2 - 1.0)
    #t = t[0]* tf.ones_like(r)
    with tf.GradientTape() as outer_r:
        outer_r.watch(r)
        with tf.GradientTape() as inner:
            inner.watch([t, r])
            f_ = f(t, r)
            c_ = c(t)
            grad = inner.gradient(f_ + c_, [t, r])
        lhs = grad[0]
        f_r = grad[1]
    f_rr = outer_r.gradient(f_r, r)
    a = (D + z*r2) * f_r
    b = 4.0*r*(z + 2.0)
    eqn = - r*lhs + a - b + D * r * (f_rr - f_r**2)
    return tf.reduce_mean(eqn**2)


# define integral objective
def integral_objective(t):
    r_ = tf.random.uniform(shape=(1000, 1), minval=0.0, maxval=R, dtype=f.dtype)
    imp_ratio = tf.exp(-f(t[0]*tf.ones_like(r_), r_)) * r_
    integral = 2.0 * np.pi * R * tf.reduce_mean(imp_ratio)
    return tf.reduce_sum((c(t) - tf.math.log(integral))**2)


# define static space time points for training
grid_size=40
t, r = tf.cast(tf.split([[t, r] for t in np.linspace(time_[0]+5e-2, time_[1], num=grid_size) for r in np.linspace(space_r[0], space_r[1],\
                num=grid_size)], [1, 1], axis=1), f.dtype)
r_1 = tf.reshape(tf.linspace(tf.cast(0.0, f.dtype), R, num=1000), shape=(-1, 1))



# joint training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for i in range(1000):
    with tf.GradientTape() as tape_f, tf.GradientTape() as tape_c:
        tape_f.watch(f.trainable_weights)
        tape_c.watch(c.trainable_weights)
        diff_loss = diff_op(t, r)
        init_loss = init_cond(tf.random.uniform(shape=(1000, 1), minval=0.0, maxval=R, dtype=f.dtype))
        integral_loss = integral_objective(c.domains[0].sample(1))
        loss = diff_loss + init_loss
        print('step = {}, losses = {}'.format(i + 1, [diff_loss.numpy(), init_loss.numpy(), integral_loss.numpy()]))
        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            break
        grads = tape_f.gradient(loss, f.trainable_weights)
        optimizer.apply_gradients(zip(grads, f.trainable_weights))
        grads = tape_c.gradient(integral_loss, c.trainable_weights)
        optimizer.apply_gradients(zip(grads, c.trainable_weights))



# define the final solution
class Sol(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=f.dtype)
    def call(self, t, x, y):
        r2 = x*x + y*y
        r = tf.sqrt(r2)
        return tf.exp(-f(t, r)- c(t))

P = Sol()


# plot and save the results
alpha = 0.4
plotter = nnp.NNPlotter(funcs=[P], space=[alpha*space_x, alpha*space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=alpha*space_x, y_lim=alpha*space_y, z_lim=None)
f.save_weights('saved models/' + model_name)
c.save_weights('saved models/' + model_name + '_norm')
print(c(tf.reshape(tf.linspace(tf.cast(0.0, f.dtype), 5.0, num=100), shape=(-1, 1))))
print('integral 1: ', quad.monte_carlo(P, [beta*space_x, beta*space_y], time=time_[0]))
print('integral 2: ', quad.monte_carlo(P, [beta*space_x, beta*space_y], time=time_[1]))
