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
import polyr

# exponential ansatz
D = 0.5
R = 6.0
alpha = 0.3
beta = 1.0
gamma = 1.0
space_r = np.array([0.0, R])
space_theta = np.array([-np.pi, np.pi])
space_x = gamma * R * np.array([-1., 1.])
space_y = gamma * R * np.array([-1., 1.])
time_ = np.array([0.0, 2.0])
num_nodes = 25
num_layers = 3
nn_type = 'LSTM'
model_name = 'cricle14_{}_{}_{}'.format(num_nodes, num_layers, str(D).replace('.', '_'))


# define f
f = polyr.PolyR(dim = 2, num_components=4, num_nodes=num_nodes, num_layers = num_layers, dtype=tf.float32)
"""
try:
   f.load_weights('saved models/' + model_name).expect_partial()
except:
   pass
#"""

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
            p_ = f(t, r)
            grad_p = inner.gradient(p_, [t, r])
        p_t = grad_p[0]
        p_r = grad_p[1]
    p_rr = outer_r.gradient(p_r, r)
    b = p_r
    a = (D + z*r2) * b
    c = 4.0*r*(z + 2.0)
    eqn = - r*p_t + a - c + D * r * (p_rr - b**2)
    return tf.reduce_mean(eqn**2)


# add as an objective
f.add_objective(diff_op)
f.add_domain(time_, 'uniform', space_r, 'uniform')
print(diff_op(*f.domains[0].sample(5)))
# define initial condition
log_4_R2 = tf.cast(2.0 * tf.math.log(2.0 * R), f.dtype)
init_cond = lambda r: tf.reduce_mean((f(tf.zeros_like(r), r) - 0.5*r*r)**2)
print(f(*f.domains[0].sample(5)))
f.summary()
#exit()

# add as an objective
f.add_objective(init_cond)
f.add_domain(space_r, 'uniform')

#"""
# learn the solution
num_samples = [1000, 1000]
sample_types = ['dynamic', 'dynamic', 'dynamic']
#sample_types = ['static', 'static', 'static']
f.solve(num_steps = 1500, num_samples = num_samples, sample_types = sample_types, initial_rate = 0.001)
f.save_weights('saved models/' + model_name)
#"""
# define the final solution
class Sol(tf.keras.models.Model):
   def __init__(self):
       super().__init__(dtype=f.dtype)
   def call(self, t, x, y):
       r = tf.sqrt(x*x + y*y)
       return tf.exp(-f(t, r))

class LogSol_inf(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=f.dtype)
    def call(self, t, r):
        return (r**2 - 1.0)**2/D

class LogSol_0(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=f.dtype)
    def call(self, t, r):
        return r**2/2.0

P = Sol()
f_inf = LogSol_0()
#"""
# plot and save the results
plotter = nnp.NNPlotter(funcs=[f, f_inf], space=[alpha*space_r], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name + '_f'), t=time_, x_lim=alpha*space_r, y_lim=None, z_lim=None)
"""
plotter = nnp.NNPlotter(funcs=[P], space=[alpha*space_x, alpha*space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=alpha*space_x, y_lim=alpha*space_y, z_lim=None, wireframe=True)
#"""
print('integral 1: ', quad.monte_carlo(P, [beta*space_x, beta*space_y], time=time_[0]))
print('integral 2: ', quad.monte_carlo(P, [beta*space_x, beta*space_y], time=time_[1]))
#"""
