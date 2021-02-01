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
D = 0.01
R = 2.0
gamma = 1.0
space_r = np.array([0.0, R])
space_theta = np.array([-np.pi, np.pi])
space_x = gamma * R * np.array([-1., 1.])
space_y = gamma * R * np.array([-1., 1.])
log_2_pi = tf.math.log(2.0 * np.pi)
time_ = [0.0, 2.0]
num_nodes = 40
num_layers = 4
nn_type = 'LSTM'
model_name = 'cricle3_{}_{}_{}'.format(num_nodes, num_layers, 'DGM')

p = dgm.DGM(dim = 2, num_nodes=num_nodes, num_layers = num_layers, final_activation=tf.keras.activations.linear, dtype=tf.float64)
p.add_domain(time_, 'uniform', space_r, 'uniform')
v = tf.cast(log_2_pi, p.dtype)
#"""
try:
    p.load_weights('saved models/' + model_name).expect_partial()
except:
    pass
#"""
print(p(*p.domains[0].sample(5)))
p.summary()
sqrt2 = tf.cast(tf.sqrt(2.0), p.dtype)
init_cond = lambda r: tf.reduce_mean((p(tf.zeros_like(r), r) - tf.sqrt(0.5*r*r+v))**2)
@ut.timer
def diff_op(t, r):
    r2 = r*r
    z = 4.0*(r2 - 1.0)
    #t = t[0]* tf.ones_like(r)
    with tf.GradientTape() as outer_r:
        outer_r.watch(r)
        with tf.GradientTape() as inner:
            inner.watch([t, r])
            p_ = p(t, r)
            grad_p = inner.gradient(p_, [t, r])
        p_t = grad_p[0]
        p_r = grad_p[1]
    p_rr = outer_r.gradient(p_r, r)
    b = p_*p_r
    a = (D + z*r2) * b
    c = 2.0*r*(z + 2.0)
    eqn = - r*p_*p_t + a - c + D * r * (p_*p_rr + p_r**2 - 2.0*b**2)
    return tf.reduce_mean(eqn**2)

# loss functions and domains
p.add_objective(diff_op, mean_square=False)
p.add_objective(init_cond, mean_square=False)
p.add_domain(space_r, 'uniform')

normalizer = dgm.DGM(dim = 1, num_nodes=50, num_layers = 4, final_activation=tf.square, dtype=p.dtype)

#"""
try:
    normalizer.load_weights('saved models/' + model_name + '_norm').expect_partial()
except:
    pass
#"""
R_ = 2.0 * R
r_ = tf.random.uniform(shape=(1000, 1), minval=0.0, maxval=R_, dtype=p.dtype)
r_2 = r_ * r_

def integral_objective(t):
    t_ = t[0] * tf.ones_like(r_)
    imp_ratio = tf.exp(-p(t_, r_)**2) * r_
    return (normalizer(t) - 2.0 * np.pi * R_ * tf.reduce_mean(imp_ratio))**2

normalizer.add_objective(integral_objective, mean_square=False)
normalizer.add_domain(time_, 'uniform')



class Sol(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=p.dtype)
    def call(self, t, x, y):
        r2 = x*x + y*y
        r = tf.sqrt(r2)
        return tf.exp(-p(t, r)**2)/normalizer(t)

P = Sol()
grid_size=30
t, r = tf.split([[t, r] for t in np.linspace(time_[0]+5e-3, time_[1], num=grid_size) for r in np.linspace(space_r[0], space_r[1],\
                num=grid_size)], [1, 1], axis=1)
p.solve(num_steps = 1000, num_samples = [grid_size**2, 1000], samples=[[t, r], [r]], sample_types = ['static', 'dynamic'], initial_rate =\
        0.001)
normalizer.solve(num_steps=1000, num_samples=[1], sample_types=['dynamic'], initial_rate=0.001)
plotter = nnp.NNPlotter(funcs=[P], space=[space_x, space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=space_x, y_lim=space_y, z_lim=None)
p.save_weights('saved models/' + model_name)
normalizer.save_weights('saved models/' + model_name + '_norm')
print('integral 1: ', quad.monte_carlo(P, [2.0*space_x, 2.0*space_y], time=time_[0]))
print('integral 2: ', quad.monte_carlo(P, [2.0*space_x, 2.0*space_y], time=time_[1]))
