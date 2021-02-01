import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import tensorflow as tf
import numpy as np
import armm_ps as ps
import armm_sp as sp
import nnplot as nnp
import dgm

r = 2.0
space_x = r * np.array([-1., 1.])
space_y = r * np.array([-1., 1.])

def annulus_params(num_pts):
    r_list = np.linspace(0.0, r, num=num_pts)
    theta_list = np.linspace(0., 2.*np.pi, num=num_pts, endpoint=False)
    mu = [(r_*np.cos(theta_), r_*np.sin(theta_)) for r_ in r_list for theta_ in theta_list]
    sigma = [(-1.0, -1.0) for i in range(len(mu))]
    return num_pts**2, [mu, sigma]

def circle_params(num_pts):
    theta = np.linspace(0., 2.*np.pi, num=num_pts, endpoint=False)
    mu = list(zip(np.cos(theta), np.sin(theta)))
    sigma = [(-1.0, -1.0) for i in range(len(mu))]
    return num_pts, [mu, sigma]

#num_components, param_inits = circle_params(25)
num_components, param_inits = annulus_params(5)

nn = dgm.DGM(num_components = num_components, dim = 3, num_nodes=1, num_layers = 2, time_dependent = True, param_inits=param_inits,\
               init = 'random_normal', nn_type='LSTM')

plotter = nnp.NNPlotter(funcs=[nn], space=[space_x, space_y], num_pts_per_dim=25)
plotter.animate(file_path='../../images/annulus_init.mp4', t=[0., 10.], num_frames=48)# num_frames=24, x_lim=space_x, y_lim=space_y, z_lim=[0.,0.2])
x = []
for i in range(nn.dim):
    x.append(tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (10, 1)), dtype=tf.float32))
print(nn(*x))
print(nn(x[1], x[1], x[2]))
#nn.save_weights('temp')
nn.summary()
"""
print(len(nn.trainable_weights))
for w in nn.trainable_weights:
    print(w.name, w.shape, w.numpy())
"""
nn.save_weights('temp/circle_init_1')
