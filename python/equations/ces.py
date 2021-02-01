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
import armm_ps as ps
import armm_sp as sp
import nnplot as nnp
import integrate as quad
import utility as ut
import dgm

# exponential ansatz
D = 1.0
R = 2.0
gamma = 100.0
space_r = np.array([0.0, R])
space_theta = np.array([-np.pi, np.pi])
space_x = gamma * R * np.array([-1., 1.])
space_y = gamma * R * np.array([-1., 1.])
log_2_pi = tf.math.log(2.0 * np.pi)

time_ = [0.0, 10.0]
num_nodes = 30
num_layers = 3
nn_type = 'LSTM'
model_name = 'exp_rs_{}_{}_{}'.format(num_nodes, num_layers, 'DGM')


p = dgm.DGM(dim = 2, num_nodes=num_nodes, num_layers = num_layers, final_activation=tf.keras.activations.exponential)
p.add_domain(time_, 'uniform', space_r, 'uniform')
#"""
try:
    p.load_weights('saved models/' + model_name).expect_partial()
except:
    pass
#"""
print(p(*p.domains[0].sample(5)))
p.summary()
init_cond = lambda x, y: tf.reduce_mean(p(tf.zeros_like(x), x, y) - tf.exp(-0.5*(x*x + y*y))/(2.*np.pi))**2
@ut.timer
def diff_op(t, r):
    r2 = r*r
    z = 4.0*(r2 - 1.0)
    with tf.GradientTape() as outer_r:
        outer_r.watch(r)
        with tf.GradientTape() as inner:
            inner.watch([t, r])
            p_ = p(t, r)
            grad_p = inner.gradient(p_, [t, r])
        p_t = grad_p[0]
        p_r = grad_p[1]
    p_rr = outer_r.gradient(p_r, r)
    a = (D + z*r2) * p_r
    c = 4.0*r*(z + 2.0)
    eqn = - r*p_t + a - c + D * r * (p_rr - p_r**2)
    return tf.reduce_mean(eqn**2) + tf.reduce_mean((p(tf.zeros_like(r), r) - 0.5*r2 - log_2_pi)**2)

# loss functions and domains
p.add_objective(diff_op, mean_square=False)
#p.add_objective(init_cond, mean_square=False)
p.add_domain(space_x, 'normal', space_y, 'normal')

def integral_objective(x, y):
    t = np.random.uniform(time_[0], time_[1], 1)[0] * tf.ones_like(x)
    p_ = tf.exp(-0.5 *(x*x + y*y)) / (2.0 * np.pi * tf.math.erf(r/tf.sqrt(2.0))**2)
    return 0.1 * (tf.reduce_mean(p(t, x, y)/p_) - 1.0)**2
#p.add_objective(integral_objective, mean_square=False)
#p.add_domain(space_x, 'truncated_normal', space_y, 'truncated_normal')
class Sol(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=p.dtype)
    def call(self, t, x, y):
        r = tf.sqrt(x*x + y*y)
        return tf.exp(-p(t, r))

P = Sol()

#p.solve(num_steps = 1500, num_samples = [1000, 1000, 1000], sample_types = ['dynamic', 'dynamic', 'dynamic'], initial_rate = 0.001)
plotter = nnp.NNPlotter(funcs=[P], space=[space_x, space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=space_x, y_lim=space_y, z_lim=None)
p.save_weights('saved models/' + model_name)
#print('integral 1: ', quad.monte_carlo(P, [2.0*space_x, 2.0*space_y], time=time_[0]))
#print('integral 2: ', quad.monte_carlo(P, [2.0*space_x, 2.0*space_y], time=time_[1]))
