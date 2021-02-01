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

alpha = 1.0
L = np.pi/alpha
space_x = np.array([0., L])
time_ = [0.0, 1.0]


#num_components, param_inits = circle_params(25)
num_nodes = 50
num_layers = 3
nn_type = 'LSTM'
model_name = 'heat_1d_{}_{}'.format(num_nodes, num_layers)


p = dgm.DGM(dim = 2, num_nodes=num_nodes, num_layers = num_layers, dtype=tf.float64)
p.add_domain(time_, 'uniform', space_x, 'uniform')
"""
try:
    p.load_weights('saved models/' + model_name).expect_partial()
except:
    pass
#"""
print(p(*p.domains[0].sample(5)))
p.summary()

@ut.timer
def diff_op(t, x):
    with tf.GradientTape() as outer_x:
        outer_x.watch(x)
        with tf.GradientTape() as inner:
            inner.watch([t, x])
            p_ = p(t, x)
            grad_p = inner.gradient(p_, [t, x])
        p_t = grad_p[0]
        p_x = grad_p[1]
    p_xx = outer_x.gradient(p_x, x)
    return tf.reduce_mean( (-p_t + p_xx/alpha)**2 )

# loss functions and domains
p.add_objective(diff_op, mean_square=False)

def bdry_cond(t, x):
    loss_1 = tf.reduce_mean(tf.square(p(tf.zeros_like(x), x) - tf.sin(alpha*x)))
    loss_2 = tf.reduce_mean(tf.square(p(t, tf.zeros_like(t))))
    loss_3 = tf.reduce_mean(tf.square(p(t, L*tf.ones_like(t))))
    return loss_1 + loss_2 + loss_3

p.add_objective(bdry_cond, mean_square=True)
p.add_domain(time_, 'uniform', space_x, 'uniform')

class Sol(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=p.dtype)
    def call(self, t, x):
        return tf.exp(-alpha*t)*tf.sin(alpha*x)

s = Sol()

# learn the solution
p.solve(num_steps = 200, num_samples = [1000, 100, 100], sample_types = ['dynamic', 'dynamic', 'dynamic'], initial_rate = 0.001)
plotter = nnp.NNPlotter(funcs=[s, p], space=[space_x], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), style='standard', t=time_, x_lim=space_x, y_lim=None, z_lim=None)
plotter = nnp.NNPlotter(funcs=[p, s], space=[space_x], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format('nn_' + model_name), style='error', t=time_, x_lim=space_x, y_lim=None, z_lim=None)
p.save_weights('saved models/' + model_name)
