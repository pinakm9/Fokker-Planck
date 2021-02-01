import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import tensorflow as tf
import numpy as np
import nnsolve2 as nn

argmmb = nn.ARGMMBlock(num_components = 5, input_dim = 5, time_dependent = False)
argmm = nn.ARGMM(num_components = 5, dim = 5, num_LSTM_layers = 2, time_dependent = False)
x = []
for i in range(5):
    x.append(tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (1, 1)), dtype=tf.float32))
print(argmmb(tf.concat(x, axis=1)))
print(argmm(tf.concat(x, axis=1)))

argmm.summary()
"""
argmmb = nn.ARGMMBlock(num_components = 5, input_dim = 5, time_dependent = True)
argmm = nn.ARGMM(num_components = 5, dim = 5, num_LSTM_layers = 1, time_dependent = True)
x = []
for i in range(5):
    x.append(tf.convert_to_tensor(np.random.uniform(low=0., high=1., size = (10, 1)), dtype=tf.float32))
print(argmmb(*x))
print(argmm(*x))
argmm.summary()
"""
