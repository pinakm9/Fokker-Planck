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
import utility as ut

r = 2.0
space_x = r * np.array([-1., 1.])
space_y = r * np.array([-1., 1.])
time_ = [0.0, 10.0]

def annulus_params(num_pts):
    r_list = np.linspace(0.1, r, num=num_pts)
    theta_list = np.linspace(-np.pi, np.pi, num=num_pts)
    mu = [(r_*np.cos(theta_), r_*np.sin(theta_)) for r_ in r_list for theta_ in theta_list]
    sigma = [(-1.0, -1.0) for i in range(len(mu))]
    return num_pts**2, [mu, sigma]

def circle_params(num_pts):
    theta = np.linspace(-np.pi, np.pi, num=num_pts)
    mu = list(zip(np.cos(theta), np.sin(theta)))
    sigma = [(-1.0, -1.0) for i in range(len(mu))]
    return num_pts, [mu, sigma]

num_components, param_inits = circle_params(2)
#num_components, param_inits = annulus_params(2)
#num_components, param_inits = 8,  None
num_nodes = 50
num_layers = 3
nn_type = 'LSTM'
model_name = 'circle_{}_{}_{}_{}_{}'.format(nn_type, num_components, num_nodes, num_layers, 'None')


p = sp.ArMmSp(num_components = num_components, dim = 3, num_nodes=num_nodes, num_layers = num_layers, time_dependent = True, param_inits=param_inits,\
              nn_type=nn_type)
p.add_domain([time_, space_x, space_y])
try:
    p.load_weights('saved models/' + model_name).expect_partial()
except:
    pass
p(*p.domain_sampler(0, 1))
p.summary()

@ut.timer
def diff_op(t, x, y):
    z = -4.0*(x*x + y*y - 1.0)
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
    a = (x*z + y) * p_x
    b = (y*z - x) * p_y
    c = (4.0*z - 8.0) * p_
    return  1.41*(p_t + a + b + c - 0.5*(p_xx + p_yy))

# loss functions and domains
p.add_objective(diff_op)
p.add_objective(lambda x, y: (p(tf.zeros_like(x), x, y) - tf.exp(-0.5*(x*x + y*y))/(2.*np.pi)))
p.add_domain([space_x, space_y])

# learn the solution
p.solve(num_steps = 500, num_samples = [1000, 1000], initial_rate = 0.0005)#, decay_steps=500, decay_rate=0.1)
plotter = nnp.NNPlotter(funcs=[p], space=[space_x, space_y], num_pts_per_dim=30)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=space_x, y_lim=space_y, z_lim=None)
p.save_weights('saved models/' + model_name)
