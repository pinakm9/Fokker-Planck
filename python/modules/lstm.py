import tensorflow as tf
from modules import nnsolver as nns
import tables

class LSTMForgetBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float32):
        super().__init__(name='LSTMForgetBlock', dtype=dtype)
        self.W_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_i', use_bias=False)
        self.U_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_i')
        self.W_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_o', use_bias=False)
        self.U_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_o')
        self.W_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_c', use_bias=False)
        self.U_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_c')

    def call(self, x, h, c):
        f = tf.keras.activations.sigmoid(self.W_f(x) + self.U_f(h))
        i = tf.keras.activations.sigmoid(self.W_i(x) + self.U_i(h))
        o = tf.keras.activations.sigmoid(self.W_o(x) + self.U_o(h))
        c_ = tf.keras.activations.tanh(self.W_c(x) + self.U_c(h))
        c = tf.keras.activations.tanh(f*c + i*c_)
        return o*c, c

class LSTMForget(nns.NNSolver):
    def __init__(self, num_nodes, num_layers, name='LSTMForget', data_path='../../data', dpl_type='h5',\
                dtype=tf.float32, final_activation=tf.keras.activations.linear):
        super().__init__(name=name, data_path=data_path, dpl_type=dpl_type, dtype=dtype)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.lstm_layers = [LSTMForgetBlock(num_nodes, dtype=dtype) for i in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(1, dtype=dtype, name='final', activation=final_activation)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)

    def call(self, *args):
        x = tf.concat(args, axis=1)
        h = tf.zeros_like(x)
        print('hiya', x.shape[0])
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_layers):
            h, c = self.lstm_layers[i](x, h, c)
            h = self.batch_norm(h)
            c = self.batch_norm(c)
        return self.final_dense(h)
"""
class PolyR(nns.NNSolver):
    def __init__(self, degree, num_nodes, num_layers, threshold, name='PolyR', dtype=tf.float32, final_activation=tf.keras.activations.linear):
        super().__init__(name=name, dtype=dtype)
        self.degree = degree
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.lstm_layers_0 = [LSTMForgetBlock(num_nodes, dtype=dtype) for i in range(num_layers)]
        self.lstm_layers_1 = [LSTMForgetBlock(num_nodes, dtype=dtype) for i in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(degree + 1, dtype=dtype, name='final', activation=final_activation)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)

    def call(self, *args):
        x = tf.concat(args[:-1], axis=1)
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_layers):
            h, c = self.lstm_layers[i](x, h, c)
            h = self.batch_norm(h)
            c = self.batch_norm(c)
        r = tf.concat([args[-1]**n for n in range(self.degree + 1)], axis=1)
        return tf.reshape(tf.reduce_sum(self.final_dense(h) * r, axis=1), shape=(-1, 1))
# Testing ...
#"""
"""
lf = LSTMForget(10, 3, dpl_type='csv')
lf.add_domain([-6., 6.], 'uniform', [-6., 6.], 'uniform')
lf.add_domain([-6., 6.], 'uniform', [-6., 6.], 'uniform')
x = lf.domains[0].sample(5)
print(x)
print(lf(*x))
lf.build_db(num_pts=[1000, 10000])
lf.dpl.open_db()
print(lf.dpl.read_db([7, 50], [9, 11]))
lf.save_weights()
#"""
"""
dp = nns.DataPipelineCSV('../../data')
dp.add_domain([-6., 6.], 'uniform', [-6., 6.], 'uniform')
dp.add_domain([-6., 6.], 'uniform', [-6., 6.], 'uniform')
dp.build_db(num_pts=[1000, 10000])
dp.open_db()
print(dp.read_db(num_pts=[50, 50], starts=[100, 555]))
print(dp.read_db(num_pts=[50, 50], starts=[100, 589]))
dp.close_db()
#"""
