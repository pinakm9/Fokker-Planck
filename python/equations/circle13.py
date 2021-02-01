# add required folders to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
image_dir = str(script_dir.parent.parent)
print(image_dir)
sys.path.insert(0, module_dir)
# import modules
import tensorflow as tf
import numpy as np
from modules import nnplot as nnp
from modules import integrate as quad
from modules import utility as ut
from modules import lstm

# exponential ansatz
D = 0.51
R = 6.0
T = 10.0
alpha = 0.3
beta = 3.0
gamma = 1.0
space_r = np.array([0.0, R])
space_theta = np.array([-np.pi, np.pi])
space_x = gamma * R * np.array([-1., 1.])
space_y = gamma * R * np.array([-1., 1.])
time_ = np.array([0.0, T])
num_nodes = 50
num_layers = 3
nn_type = 'LSTMForget'
model_name = 'cricle13_{}_{}_{}_{}'.format(num_nodes, num_layers, nn_type, str(D).replace('.', '_'))


# define f
f = lstm.LSTMForget(name=model_name, num_nodes=num_nodes, num_layers = num_layers, dtype=tf.float32, final_activation=tf.square)
#"""
try:
   f.load_weights()
except:
   pass
#"""

# define differential operator
#@ut.timer
def diff_op(t, r):
    r2 = r*r
    z = 4.0*(r2 - 1.0)
    #t = t[0]* tf.ones_like(r)
    with tf.GradientTape() as outer_r:
        outer_r.watch(r)
        with tf.GradientTape() as inner:
            inner.watch([t, r])
            f_ = f(t, r)
        grad_f = inner.gradient(f_, [t, r])
        f_t = grad_f[0]
        f_r = grad_f[1]
    f_rr = outer_r.gradient(f_r, r)
    b = f_r
    a = (D + z*r2) * b
    c = 4.0*r*(z + 2.0)
    eqn = - r*f_t + a - c + D * r * (f_rr - b**2)
    return tf.reduce_mean(eqn**2)

# add as an objective
f.add_objective(diff_op)
f.add_domain(time_, 'uniform', space_r, 'uniform')

# define initial condition
log_4_R2 = tf.cast(2.0 * tf.math.log(2.0 * R), f.dtype)
init_cond = lambda r: 10.0*tf.reduce_mean((f(tf.zeros_like(r), r) - 0.5*r*r)**2)
#f.summary()
#exit()

# add as an objective
f.add_objective(init_cond)
f.add_domain(space_r, 'uniform')

# define boundary condition
def bdry_cond(t):
    r = R*tf.ones_like(t)
    with tf.GradientTape() as tape:
        tape.watch(r)
        p = tf.exp(-f(t, r))
    dp_dr = tape.gradient(p, r)
    expr = 4.0*R*(R**2-1.0)*p + D*R*dp_dr
    return tf.reduce_mean(expr**2)

# add as an objective
f.add_objective(bdry_cond)
f.add_domain(time_, 'uniform')
#"""
# learn the solution
num_samples = [10000, 10000, 10000]
f.build_db(num_samples, normalize=False)
f.solve(epochs = 1000, num_batches=10, initial_rate = 0.001)
f.save_weights()
#"""
# define the final solution
class Sol(tf.keras.models.Model):
   def __init__(self):
       super().__init__(dtype=f.dtype)
   def call(self, t, x, y):
       r = tf.sqrt(x*x + y*y)
       return tf.exp(-f(t, r))

class Sol_(tf.keras.models.Model):
   def __init__(self):
       super().__init__(dtype=f.dtype)
   def call(self, t, r):
       return r*tf.exp(-f(t, r))

class LogSol_inf(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=f.dtype)
    def call(self, t, r):
        return (r**2 - 1.0)**2/D

class LogSol(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=f.dtype)
    def call(self, t, r):
        return f(*f.normalize(0, t, r))

P = Sol()
P_ = Sol_()
f_inf = LogSol_inf()
f_ = LogSol()
#"""
# plot and save the results
plotter = nnp.NNPlotter(funcs=[f], space=[space_r], num_pts_per_dim=40)
#plotter.animate(file_path='../../images/{}.mp4'.format(model_name + '_f'), t=time_, x_lim=space_r, y_lim=None, z_lim=None)
#"""
plotter = nnp.NNPlotter(funcs=[P], space=[alpha*space_x, alpha*space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=alpha*space_x, y_lim=alpha*space_y, z_lim=None, wireframe=True)
#"""
print('integral 1: ', quad.monte_carlo(P_, [space_r], time=time_[0]))
print('integral 2: ', quad.monte_carlo(P_, [space_r], time=time_[1]))
#"""
