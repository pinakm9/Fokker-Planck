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

D = 5.0
log_C = tf.math.log( 0.5 * tf.sqrt(D * np.pi**3) * (1.0 + tf.math.erf(1.0/tf.sqrt(D) ) ))
log_2_pi = tf.math.log(2.*np.pi)
print('log_C is {}'.format(log_C))
r = 2.0
space_x = r * np.array([-1., 1.])
space_y = r * np.array([-1., 1.])
time_ = [0.0, 10.0]


#num_components, param_inits = circle_params(25)
num_nodes = 57
num_layers = 5
nn_type = 'LSTM'
model_name = 'cbk_{}_{}'.format(num_nodes, num_layers)


p = dgm.DGM(dim = 3, num_nodes=num_nodes, num_layers = num_layers, regularizer=tf.keras.regularizers.L2(l2=0.01))
p.add_domain([[1e-3, 15.0], space_x, space_y])
#"""
try:
    p.load_weights('saved models/' + model_name).expect_partial()
except:
    pass
#"""
print(p(*p.domain_sampler(0, 5)))
p.summary()

@ut.timer
def diff_op(t, x, y):
    z = 4.0*(x*x + y*y - 1.0)
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
    a = (x*z) * p_x
    b = (y*z) * p_y
    return   p_t + a + b - D * (p_xx + p_yy)

# loss functions and domains
p.add_objective(diff_op, mean_square=True)
print(diff_op(*p.domain_sampler(0, 5)))

def init_cond(x, y):
    r2 = x*x + y*y
    log_init_p = -0.5*r2 - log_2_pi
    log_steady_p = -(r2 - 1.0)**2/D - log_C
    t0 = tf.fill(x.shape, time_[0])
    #print(log_init_p, log_steady_p)
    return p(t0, x, y) - tf.exp(log_init_p / log_steady_p)



p.add_objective(init_cond, mean_square=True)
p.add_domain([space_x, space_y])
print(init_cond(*p.domain_sampler(1, 5)))


# create the full solution
class Sol(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=p.dtype)
    def call(self, t, x, y):
        r2 = x*x + y*y
        log_steady_p = -(r2 - 1.0)**2/D - log_C
        return p(t, x, y) * tf.exp(log_steady_p)

P = Sol()


# learn the solution
#print('integral 1: ', quad.monte_carlo(p, [space_x, space_y], time=time_[0]))
p.solve(num_steps = 2000, num_samples = [1000, 1000, 1000], sample_types = ['static', 'dynamic'], initial_rate = 0.001)
plotter = nnp.NNPlotter(funcs=[P], space=[space_x, space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=space_x, y_lim=space_y, z_lim=None)
plotter = nnp.NNPlotter(funcs=[p], space=[space_x, space_y], num_pts_per_dim=40)
plotter.animate(file_path='../../images/{}.mp4'.format('nn_' + model_name), t=time_, x_lim=space_x, y_lim=space_y, z_lim=None)
p.save_weights('saved models/' + model_name)
#print('integral 2: ', quad.monte_carlo(p, [space_x, space_y], time=time_[1]))
