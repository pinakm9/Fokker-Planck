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

r = 5.0
space_x = r * np.array([-1., 1.])
space_y = r * np.array([-1., 1.])
time_ = [0.0, 10.0]

armmb = sp.ArMmSpBlock(dim = 2, time_dependent = False)
mu = [(-2.,), (2.,), (-4.,)]
sigma = [(0.,), (0.,), (0.,)]

nn = sp.ArMmSp(num_components = 3, dim = 7, num_nodes=1, num_layers = 1, time_dependent = True, param_inits=None)
#print(nn.armmbs[0].__dict__['param_0_0'])#, 'param_0_0' + '_b_f'))
"""
plotter = nnp.NNPlotter(funcs=[armmb], space=[space_x], num_pts_per_dim=100)
plotter.plot(file_path='../../images/ArMmSpBlock1d.png')
#"""


x = []
for i in range(armmb.dim):
    x.append(tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (10, 1)), dtype=tf.float32))
print(armmb(*x))
x = []
for i in range(nn.dim):
    x.append(tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (10, 1)), dtype=tf.float32))
print(nn.armmbs[0](*x))
print(nn.armmbs[1](*x))
print(nn(*x))
"""
plotter = nnp.NNPlotter(funcs=[nn], space=[space_x, space_y], num_pts_per_dim=50)
plotter.plot(file_path='../../images/armmsp_circleNN.png')
#"""
# integration test
nn.add_domain([time_, space_x, space_y])
num_pts = 10000
t, x, y = nn.domain_sampler(0, num_pts)
print(tf.math.reduce_sum(nn(tf.ones_like(x), x, y)))
