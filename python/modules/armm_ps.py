import tensorflow as tf
import nnsolver as nns
import build_layers as bl
import density as ds

class ArMmPsBlock(tf.keras.layers.Layer):
    def __init__(self, dim, density = ds.Normal(), time_dependent = True, num_components = 5, num_nodes=50, num_LSTM_layers=3,\
                 activation=tf.keras.activations.tanh, dtype=tf.float32):
        super(ArMmPsBlock, self).__init__(dtype=dtype)
        self.num_components = num_components
        self.dim = dim
        self.density = density
        self.num_nodes = num_nodes
        self.num_LSTM_layers = num_LSTM_layers
        self.activation = activation
        self.time_dependent = time_dependent
        if self.dim == 1:
            self.time_dependent = False

    def build(self, input_shape):
        if self.dim > 1:
            for i in range(self.num_components):
                for j in range(self.density.num_params):
                    bl.build_LSTM(obj=self, nn_name='param_' + str(i) + '_' + str(j), input_dim=self.dim - 1, output_dim=1,\
                               num_nodes=self.num_nodes, num_LSTM_layers=self.num_LSTM_layers)
                bl.build_LSTM(obj=self, nn_name='c_' + str(i), input_dim=self.dim - 1, output_dim=1,\
                           num_nodes=self.num_nodes, num_LSTM_layers=self.num_LSTM_layers)
        else:
            for i in range(self.num_components):
                for j in range(self.density.num_params):
                    setattr(self, 'param_' + str(i) + '_' + str(j), self.add_weight(shape=(1,), initializer="zeros", trainable=True))
                setattr(self, 'c_' + str(i), self.add_weight(shape=(1,), initializer="random_normal", trainable=True))

    def call(self, *args):
        probs = []
        coeffs = []
        if self.dim > 1:
            input = tf.concat(args[:-1], 1)
            for i in range(self.num_components):
                # compute parameters
                params = []
                for j in range(self.density.num_params):
                    params.append(bl.call_LSTM(obj=self, nn_name='param_' + str(i) + '_' + str(j), input=input,\
                                               num_LSTM_layers=self.num_LSTM_layers,activation=self.activation))
                c = bl.call_LSTM(obj=self, nn_name='c_' + str(i), input=input, num_LSTM_layers=self.num_LSTM_layers,\
                                 activation=self.activation)
                # make parameters admissible
                params = bl.make_admissible_params(params, self.density.param_details)
                # collect probabilities and coefficients
                probs.append(self.density.pdf(args[-1], *params))
                coeffs.append(c)

        else:
            for i in range(self.num_components):
                params = []
                for j in range(self.density.num_params):
                    params.append(tf.fill(args[0].shape, getattr(self, 'param_' + str(i) + '_' + str(j)) ))
                c = tf.fill(args[0].shape, getattr(self, 'c_' + str(i)) )
                # make parameters admissible
                params = bl.make_admissible_params(params, self.density.param_details)
                # collect probabilities and coefficients
                probs.append(self.density.pdf(args[-1], *params))
                coeffs.append(c)
        print('params', params, 'coeffs', coeffs)
        probs = tf.convert_to_tensor(probs, dtype=self.dtype)
        coeffs = tf.keras.activations.softmax(tf.convert_to_tensor(coeffs, dtype=self.dtype), axis=0)
        return tf.math.reduce_sum(coeffs * probs, axis=0)

class ArMmPs(nns.NNSolver):
    """
    Description:
        Class for implementing autoregressive mixture model
    """
    def __init__(self, dim, density = ds.Normal(), num_components=5, num_nodes=50, num_LSTM_layers=1, time_dependent=True,\
                 activation=tf.keras.activations.tanh, name='ArMmPs_Model', dtype=tf.float32):
        super(ArMmPs, self).__init__(dim=dim, name=name, dtype=dtype)
        self.num_components = num_components
        self.density = density
        self.num_nodes = num_nodes
        self.num_LSTM_layers = num_LSTM_layers
        self.activation = activation
        self.time_dependent = time_dependent
        if self.dim == 1:
            self.time_dependent = False
        self.armmb = ArMmPsBlock(dim=self.dim, density=self.density, num_components=self.num_components,\
                                 num_nodes=self.num_nodes, num_LSTM_layers=self.num_LSTM_layers, activation=self.activation, dtype=dtype)
        # build the ArMmPsBlock
        self.armmb(*self.domain_sampler(domain=[[0., 1.] for i in range(self.dim)], num_samples=1))
        if not self.time_dependent:
            self.armmb_0 = ArMmPsBlock(dim=1, density=self.density, num_components=self.num_components,\
                                       num_nodes=self.num_nodes, num_LSTM_layers=self.num_LSTM_layers, activation=self.activation, dtype=dtype)
            self.armmb_0(*self.domain_sampler(domain=[[0., 1.]], num_samples=1))

    def call(self, *args):
        last_space_dim = len(args) - 1
        first_space_dim = 1 if self.time_dependent else 0
        prod_prob = 1.
        for d in range(last_space_dim, first_space_dim - 1, -1):
            probs = []
            coeffs = []
            c_sum = 0.
            if d > 0:
                input = tf.concat(args[: d], 1)
                for i in range(self.num_components):
                    params = []
                    for j in range(self.density.num_params):
                        params.append(bl.call_LSTM(obj=self.armmb, nn_name='param_' + str(i) + '_' + str(j), input=input,\
                                                   num_LSTM_layers=self.num_LSTM_layers,activation=self.activation))
                    c = bl.call_LSTM(obj=self.armmb, nn_name='c_' + str(i), input=input, num_LSTM_layers=self.num_LSTM_layers,\
                                     activation=self.activation)
                    # make parameters admissible
                    params = bl.make_admissible_params(params, self.density.param_details)
                    # collect probabilities and coefficients
                    probs.append(self.density.pdf(args[-d], *params))
                    coeffs.append(c)
            else:
                for i in range(self.num_components):
                    params = []
                    for j in range(self.density.num_params):
                        params.append(getattr(self.armmb_0, 'param_' + str(i) + '_' + str(j) ))
                    c = getattr(self.armmb_0, 'c_' + str(i))
                    # make parameters admissible
                    params = bl.make_admissible_params(params, self.density.param_details)
                    # collect probabilities and coefficients
                    probs.append(self.density.pdf(args[-1], *params))
                    coeffs.append(c)
            probs = tf.convert_to_tensor(probs, dtype=self.dtype)
            coeffs = tf.keras.activations.softmax(tf.convert_to_tensor(coeffs, dtype=self.dtype), axis=0)
            prod_prob *= tf.reduce_mean(coeffs * probs, axis=0)
        return prod_prob


class armmb_F(nns.NNSolver):
    """
    Description:
        Class for implementing autoregressive Gaussian mixture model
    """
    def __init__(self, dim, num_components=5, num_nodes=50, num_LSTM_layers=1, time_dependent=True, activation=tf.keras.activations.tanh,\
                 name='armmb_Model', dtype=tf.float32):
        super(armmb_F, self).__init__(dim=dim, name=name, dtype=dtype)
        self.num_components = num_components
        self.num_nodes = num_nodes
        self.num_LSTM_layers = num_LSTM_layers
        self.activation = activation
        self.time_dependent = time_dependent
        if self.dim == 1:
            self.time_dependent = False
        self.armmbbs = []
        if not self.time_dependent:
            self.armmbbs.append(ArMmPsBlock(dim=1, num_components=self.num_components, num_nodes=self.num_nodes,\
                                           num_LSTM_layers=self.num_LSTM_layers, activation=self.activation, dtype=dtype))
            self.armmbbs[0](*self.domain_sampler(domain=[[0., 1.]], num_samples=1))
        self.armmbbs += [ArMmPsBlock(dim=d, num_components=self.num_components, num_nodes=self.num_nodes,\
                                   num_LSTM_layers=self.num_LSTM_layers, activation=self.activation, dtype=dtype) for d in range(2, self.dim + 1)]
        # build the ArMmPsBlocks
        for block in self.armmbbs:
            block(*self.domain_sampler(domain=[[0., 1.] for j in range(block.dim)], num_samples=1))

    def call(self, *args):
        prod_prob = 1.
        for d, block in enumerate(self.armmbbs):
            prod_prob *= block(*args[: block.dim])
        return prod_prob
