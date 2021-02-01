import tensorflow as tf
import nnsolver as nns
import build_layers as bl

class PolyR(nns.NNSolver):
    """
    Description:
        Class for implementing autoregressive mixture model
    """
    def __init__(self, dim, num_components=5, num_nodes=50, num_layers=1, time_dependent=True, regularizer=None,\
                 activation=tf.keras.activations.tanh, name='PolyR_Model', dtype=tf.float32):
        super(PolyR, self).__init__(dim=dim, name=name, dtype=dtype)
        self.dim = dim
        self.num_components = num_components
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.regularizer = regularizer
        self.activation = activation
        bl.build_type_1(obj=self, nn_name='power', input_dim=1, output_dim=self.num_components,\
                      num_nodes=self.num_nodes, num_layers=self.num_layers,\
                      regularizer=self.regularizer, b_f_initializer=None)
        bl.build_type_1(obj=self, nn_name='coefficient', input_dim=self.dim - 1, output_dim=self.num_components,\
                      num_nodes=self.num_nodes, num_layers=self.num_layers,\
                      regularizer=self.regularizer, b_f_initializer=None)
    def call(self, *args):
        power = bl.call_type_1(obj=self, nn_name='power', input=args[0], num_layers=self.num_layers,\
                     activation=self.activation, final_activation=tf.square)
        coeff = bl.call_type_1(obj=self, nn_name='coefficient', input=tf.concat(args[:-1], axis=1), num_layers=self.num_layers,\
                     activation=self.activation)
        return tf.reshape(tf.reduce_sum(coeff * tf.pow(args[-1], power), axis=1), shape=(-1, 1))




class AugR(nns.NNSolver):
    """
    Description:
        Class for implementing autoregressive mixture model
    """
    def __init__(self, dim, degree=5, num_nodes=50, num_layers=1, time_dependent=True, regularizer=None,\
                 activation=tf.keras.activations.tanh, name='AugR_Model', dtype=tf.float32):
        super(AugR, self).__init__(dim=dim, name=name, dtype=dtype)
        self.dim = dim
        self.degree = degree
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.regularizer = regularizer
        self.activation = activation
        bl.build_LSTM(obj=self, nn_name='nn', input_dim=self.dim + self.degree - 1, output_dim=1,\
                      num_nodes=self.num_nodes, num_layers=self.num_layers,\
                      regularizer=self.regularizer, b_f_initializer=None)

    def call(self, *args):
        r = tf.concat([args[-1]**n for n in range(2, self.degree + 1)], axis=1)
        args = tf.concat(args, axis=1)
        input = tf.math.l2_normalize(tf.concat([args, r], axis=1))
        return bl.call_LSTM(obj=self, nn_name='nn', input=input, num_layers=self.num_layers,\
                     activation=self.activation)
