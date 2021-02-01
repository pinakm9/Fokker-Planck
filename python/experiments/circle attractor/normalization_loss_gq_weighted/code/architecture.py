# add required folders to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent.parent.parent)
sys.path.insert(0, module_dir)

import tensorflow as tf
from modules import nnsolver as nns

import tensorflow as tf
from modules import nnsolver as nns
from modules import build_layers as bl
import numpy as np

class DGM(nns.NNSolver):
    """
    Description:
        Class for implementing the DGM architechture
    """
    def __init__(self, dim, data_path, dpl_type='csv', num_nodes=50, num_layers=3, nn_type='LSTM', regularizer=None, activation=tf.keras.activations.tanh,\
                 final_activation=tf.keras.activations.linear, name='DGM_Model', dtype=tf.float32):
        super(DGM, self).__init__(name=name, data_path=data_path, dpl_type=dpl_type, dtype=dtype)
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.activation = activation
        self.nn_type = nn_type
        self.build_nn = getattr(bl, 'build_' + self.nn_type)
        self.call_nn = getattr(bl, 'call_' + self.nn_type)
        self.regularizer = regularizer
        self.final_activation = final_activation
        self.build_nn(obj=self, nn_name='nn', input_dim=self.dim, output_dim=1,\
                      num_nodes=self.num_nodes, num_layers=self.num_layers, initializer='random_normal',\
                      regularizer=self.regularizer, b_f_initializer=None)

    def call(self, *args):
        return self.call_nn(obj=self, nn_name='nn', input=tf.concat(args, 1), num_layers=self.num_layers, activation=self.activation,\
                           final_activation=self.final_activation)
