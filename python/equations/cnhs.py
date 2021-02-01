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

D = 0.5
log_C = tf.math.log( 0.5 * tf.sqrt(D * np.pi**3) * (1.0 + tf.math.erf(1.0/tf.sqrt(D) ) ))
log_2_pi = tf.math.log(2.*np.pi)
alpha = 1.0
r = 2.0
space_x = r * np.array([-1., 1.])
space_y = r * np.array([-1., 1.])
time_ = [0.0, 10.0]


#num_components, param_inits = circle_params(25)
num_nodes = 50
num_layers = 4
nn_type = 'LSTM'
model_name = 'cnhs_{}_{}'.format(num_nodes, num_layers)


p = dgm.DGM(dim = 3, num_nodes=num_nodes, num_layers = num_layers,\
            final_activation=tf.keras.activations.sigmoid)#regularizer=tf.keras.regularizers.L2(l2=0.01))
p.add_domain([time_, space_x, space_y])
"""
try:
    p.load_weights('saved models/' + model_name).expect_partial()
except:
    pass
#"""
print(p(*p.domain_sampler(0, 5)))
p.summary()

@ut.timer
def diff_op(t, x, y):
    r2 = x*x + y*y
    z = 4.0 * (r2 - 1.0)
    q = 4.0 * (z + 2.0)
    with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
        outer_x.watch(x)
        outer_y.watch(y)
        with tf.GradientTape() as inner:
            inner.watch([t, x, y])
            p_ = p(t, x, y)
            grad_p = inner.gradient(p_, [t, x, y])
        p_t = grad_p[0]
        p_x = grad_p[1]
        p_y = grad_p[2]
    p_xx = outer_x.gradient(p_x, x)
    p_yy = outer_y.gradient(p_y, y)
    Lp = (x*z) * p_x + (y*z) * p_y + q*p_ + D*(p_xx + p_yy)
    p1 = tf.exp(-(r2 - 1.0)**2/D - log_C)
    Lp1 = p1 * (r2*z*z - q)
    s = tf.exp(-alpha * t)
    return tf.reduce_mean((s * (p_t - Lp) + s * alpha * (p1 - p_) - (1.0 - s) * Lp1)**2)  #+ tf.reduce_mean(tf.exp(-p_t))

# loss functions and domains
p.add_objective(diff_op, mean_square=False)
print(diff_op(*p.domain_sampler(0, 5)))

def init_cond(x, y):
    r2 = x*x + y*y
    log_init_p = -0.5*r2 - log_2_pi
    t0 = tf.fill(x.shape, time_[0])
    #print('init', tf.exp(log_init_p - log_steady_p), '\np', p(t0, x, y))
    return p(t0, x, y) - tf.exp(log_init_p)



p.add_objective(init_cond, mean_square=True)
p.add_domain([space_x, space_y])

# create the full solution
class Sol(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=p.dtype)
    def call(self, t, x, y):
        r2 = x*x + y*y
        p1 = tf.exp(-(r2 - 1.0)**2/D - log_C)
        s = tf.exp(-alpha*t)
        return s * p(t, x, y) + (1.0 - s) * p1

P = Sol()
#exit()

def integral_objective(x, y):
    V = 4.0 * r**2
    t = tf.fill(x.shape, np.random.uniform(time_[0], time_[1]))
    return 1.0 * (tf.reduce_mean(p(t, x, y)) * V - 1.0)**2
p.add_objective(integral_objective, mean_square=False)
p.add_domain([space_x, space_y])

# learn the solution
print('integral 1: ', quad.monte_carlo(p, [space_x, space_y], time=time_[0]))
p.solve(num_steps = 200, num_samples = [1000, 1000, 1000], sample_types = ['dynamic', 'dynamic', 'dynamic'], initial_rate = 0.0005)
plotter = nnp.NNPlotter(funcs=[P], space=[space_x, space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=space_x, y_lim=space_y, z_lim=None)
plotter = nnp.NNPlotter(funcs=[p], space=[space_x, space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format('nn_' + model_name), t=time_, x_lim=space_x, y_lim=space_y, z_lim=None)
p.save_weights('saved models/' + model_name)
print('integral 2: ', quad.monte_carlo(p, [space_x, space_y], time=time_[1]))
