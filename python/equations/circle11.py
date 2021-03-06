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
R = 2.0
alpha = 1.0
beta = 1.0
gamma = 1.0
space_r = np.array([0.0, R])
space_theta = np.array([-np.pi, np.pi])
space_x = gamma * R * np.array([-1., 1.])
space_y = gamma * R * np.array([-1., 1.])
time_ = np.array([0.0, 10.0])
num_nodes = 20
num_layers = 3
nn_type = 'LSTM'
model_name = 'cricle11_{}_{}_{}'.format(num_nodes, num_layers, str(D).replace('.', '_'))


# define f
f = dgm.DGM(dim = 3, num_nodes=num_nodes, num_layers = num_layers, dtype=tf.float32)
#"""
try:
   f.load_weights('saved models/' + model_name).expect_partial()
except:
   pass
#"""

# define differential operator
@ut.timer
def diff_op(t, x, y):
    r2 = x*x + y*y
    z = 4.0*(r2 - 1.0)
    with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
        outer_x.watch(x)
        outer_y.watch(y)
        with tf.GradientTape() as inner:
            inner.watch([t, x, y])
            f_ = f(t, x, y)
        grad = inner.gradient(f_, [t, x, y])
        f_t = grad[0]
        f_x = grad[1]
        f_y = grad[2]
    f_xx = outer_x.gradient(f_x, x)
    f_yy = outer_y.gradient(f_y, y)
    a = (x*z) * f_x
    b = (y*z) * f_y
    c = 4.0*(z + 2.0)
    eqn = - f_t + a + b - c + D * (f_xx + f_yy - f_x**2 - f_y**2)
    return tf.reduce_mean(eqn**2)

# add as an objective
f.add_objective(diff_op)
f.add_domain(time_, 'uniform', space_x, 'uniform', space_y, 'uniform')

# define initial condition
log_4_R2 = tf.cast(2.0 * tf.math.log(2.0 * R), f.dtype)
init_cond = lambda x, y: 100.0 * tf.reduce_mean(( tf.exp(-f(tf.zeros_like(x), x, y)) - tf.exp(-(x*x + y*y))/(np.pi) )**2)
print(f(*f.domains[0].sample(5)))
f.summary()


# add as an objective
f.add_objective(init_cond)
f.add_domain(space_x, 'uniform', space_y, 'uniform')

# define boundary conditions
delta = 3.0
R_ = delta * R
def bdry_cond(t, theta):
    cos = tf.cos(theta)
    sin = tf.sin(theta)
    x = R_ * cos
    y = R_ * sin
    z = 4.0 * (R_**2 - 1.0)
    with tf.GradientTape() as tape:
        tape.watch([x, y])
        p_ = tf.exp(-f(t, x, y))
    grad = tape.gradient(p_, [x, y])
    p_r = grad[0] * cos + grad[1] * sin
    term_1 = tf.reduce_mean(p_r**2)
    term_2 = tf.reduce_mean((z*p_)**2)
    return term_1 + term_2

# add as an objective
f.add_objective(bdry_cond)
f.add_domain(time_, 'uniform', space_theta, 'uniform')


# learn the solution
num_samples = [1000, 1000, 100]
sample_types = ['dynamic', 'dynamic', 'dynamic']
#sample_types = ['static', 'static', 'static']
f.solve(num_steps = 5000, num_samples = num_samples, sample_types = sample_types, initial_rate = 0.001)
f.save_weights('saved models/' + model_name)

# define the final solution
class Sol(tf.keras.models.Model):
   def __init__(self):
       super().__init__(dtype=f.dtype)
   def call(self, t, x, y):
       return tf.exp(-f(t, x, y))

P = Sol()


# plot and save the results
plotter = nnp.NNPlotter(funcs=[P], space=[alpha*space_x, alpha*space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=alpha*space_x, y_lim=alpha*space_y, z_lim=None, wireframe=True)
print('integral 1: ', quad.monte_carlo(P, [beta*space_x, beta*space_y], time=time_[0]))
print('integral 2: ', quad.monte_carlo(P, [beta*space_x, beta*space_y], time=time_[1]))
