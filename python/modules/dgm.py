import tensorflow as tf
from modules import nnsolver as nns
from modules import build_layers as bl
import numpy as np

class DGM(nns.NNSolver):
    """
    Description:
        Class for implementing the DGM architechture
    """
    def __init__(self, dim, num_nodes=50, num_layers=3, nn_type='LSTM', regularizer=None, activation=tf.keras.activations.tanh,\
                 final_activation=tf.keras.activations.linear, name='DGM_Model', dtype=tf.float32):
        super(DGM, self).__init__(name=name, dtype=dtype)
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



class BoxBump:
    def __init__(self, box):
        self.box = box

    def __call__(self, *args):
        prod = 1.
        for arg in args:
            prod *= tf.exp(-1.0 / (1 - (self.box[1] - arg) * (arg - self.box[0])))
            return prod

class DGM_Tan(nns.NNSolver):
    """
    Description:
        Class for implementing the DGM architechture
    """
    def __init__(self, dim, num_nodes=50, num_layers=3, nn_type='LSTM', regularizer=None, activation=tf.keras.activations.tanh,\
                 final_activation=tf.keras.activations.linear, name='DGM_Model', dtype=tf.float32):
        super(DGM, self).__init__(dim=dim, name=name, dtype=dtype)
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
        tan_args = [tf.tan(0.5 *np.pi * arg) for arg in args[1:]]
        tan_args.append(0, args[0])
        return self.call_nn(obj=self, nn_name='nn', input=tf.concat(tan_args, 1), num_layers=self.num_layers, activation=self.activation,\
                           final_activation=self.final_activation)
