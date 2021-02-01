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

argmmb = ps.ArMmPsBlock(num_components = 20, dim = 1, time_dependent = True)
"""
argmm = ps.ArMmPs(num_components = 4, dim = 3, num_nodes=1, num_LSTM_layers = 1, time_dependent = True)
"""
"""
x = tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (10, 1)), dtype=tf.float32)
y = tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (10, 1)), dtype=tf.float32)


plotter = nnp.NNPlotter(funcs=[argmmb], space=[space_x], num_pts_per_dim=100)
plotter.plot(file_path='../../images/ArMmPsBlock1d.png', t = 9.0)
#"""


x = []
for i in range(1):
    x.append(tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (10, 1)), dtype=tf.float32))
print(argmmb(*x))















"""
#plot.plot_nn_2d(nn_sol=argmmb, space=[space_x, space_y], t=10., folder='../../images', num_pts=15)
#plot.plot_nn_2d(nn_sol=argmm.armmb, space=[space_x, space_y], t=10., folder='../../images', num_pts=15, name='block in nn')
#plot.plot_nn_2d(nn_sol=argmmb, space=[space_x, space_y], t=10., folder='../../images', num_pts=15, name='full nn')

argmmb = sp.ArMmSpBlock(input_dim = 5, time_dependent = True)
argmm = sp.ArMmSp(num_components = 5, dim = 5, num_LSTM_layers = 1, time_dependent = True)
x = []
for i in range(5):
    x.append(tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (10, 1)), dtype=tf.float32))
print(argmmb(*x))
print(argmm(*x))
argmm.summary()
"""
