import tensorflow as tf
import nnsolver as nns
import build_layers as bl
import density as ds
import numpy as np

def param_init_dim_splitter(params_list):
    num_params, num_components, dim = tf.shape(params_list)
    inits = np.empty((num_components, num_params, dim), dtype=object)
    for i in range(num_components):
        for j in range(num_params):
            for d in range(dim):
                inits[i][j][d] = tf.keras.initializers.Constant(value=params_list[j][i][d])
    return inits



class ArMmSpBlock(tf.keras.layers.Layer):
    def __init__(self, dim, density = ds.Normal(), time_dependent = True, num_nodes=50, num_layers=3, param_inits=None, init='random_normal',\
                 nn_type='LSTM', activation=tf.keras.activations.tanh, dtype=tf.float32):
        super(ArMmSpBlock, self).__init__(dtype=dtype)
        self.dim = dim
        self.density = density
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.activation = activation
        self.time_dependent = time_dependent
        if self.dim == 1:
            self.time_dependent = False
        self.space_dim = self.dim - 1 if self.time_dependent else self.dim
        if param_inits is not None:
            self.param_inits = param_inits
        else:
            self.param_inits = [[bl.ParamInitUniform() for d in range(self.space_dim)]\
                                for j in range(self.density.num_params)]
        self.nn_type = nn_type
        self.build_nn = getattr(bl, 'build_' + self.nn_type)
        self.call_nn = getattr(bl, 'call_' + self.nn_type)
        self.initializer = init

    def build(self, input_shape):
        if self.time_dependent:
            for d in range(self.space_dim):
                for j in range(self.density.num_params):
                    self.build_nn(obj=self, nn_name='param_' + str(j) + '_' + str(d), input_dim=d + 1, output_dim=1,\
                                  num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                                  b_f_initializer=self.param_inits[j][d])
        else:
            for j in range(self.density.num_params):
                name = 'param_' + str(j) + '_' + str(0)
                setattr(self, name, self.add_weight(name=name, shape=(1,), initializer=self.param_inits[j][0], trainable=True))
            for d in range(1, self.space_dim):
                for j in range(self.density.num_params):
                    self.build_nn(obj=self, nn_name='param_' + str(j) + '_' + str(d), input_dim=d, output_dim=1,\
                                  num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                                  b_f_initializer=self.param_inits[j][d])

    def call(self, *args):
        prob = 1.0
        if self.time_dependent:
            for d in range(self.space_dim):
                params = []
                for j in range(self.density.num_params):
                    params.append(self.call_nn(obj=self, nn_name='param_' + str(j) + '_' + str(d), input=tf.concat(args[: d+1], 1),\
                                               num_layers=self.num_layers, activation=self.activation))
                params = bl.make_admissible_params(params, self.density.param_details)
                prob *= self.density.pdf(args[d+1], *params)

        else:
            for d in range(self.space_dim):
                params = []
                for j in range(self.density.num_params):
                    if d > 0:
                        params.append(self.call_nn(obj=self, nn_name='param_' + str(j) + '_' + str(d), input=tf.concat(args[: d], 1),\
                                                   num_layers=self.num_layers, activation=self.activation))
                    else:
                        params.append(tf.fill(args[0].shape, getattr(self, 'param_' + str(j) + '_' + str(d) )))
                params = bl.make_admissible_params(params, self.density.param_details)
                #print(params)
                prob *= self.density.pdf(args[d], *params)
        return prob


class ArMmSp(nns.NNSolver):
    """
    Description:
        Class for implementing autoregressive Gaussian mixture model
    """
    def __init__(self, dim, density = ds.Normal(), num_components=5, num_nodes=50, num_layers=1, time_dependent=True,\
                 param_inits=None, init='random_normal', nn_type='LSTM', activation=tf.keras.activations.tanh, name='ArMmSp_Model', dtype=tf.float32):
        super(ArMmSp, self).__init__(dim=dim, name=name, dtype=dtype)
        self.num_components = num_components
        self.density = density
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.activation = activation
        self.time_dependent = time_dependent
        if self.dim == 1:
            self.time_dependent = False
        self.nn_type = nn_type
        self.build_nn = getattr(bl, 'build_' + self.nn_type)
        self.call_nn = getattr(bl, 'call_' + self.nn_type)
        self.initializer = init
        if param_inits is None:
            self.armmbs = [ArMmSpBlock(dim=self.dim, density=self.density, num_nodes=self.num_nodes,\
                                       time_dependent=self.time_dependent,num_layers=self.num_layers, param_inits=None, init = self.initializer,\
                                       nn_type = self.nn_type, activation=self.activation, dtype=dtype) for i in range(self.num_components)]
        else:
            inits = param_init_dim_splitter(param_inits)
            self.armmbs = [ArMmSpBlock(dim=self.dim, density=self.density, num_nodes=self.num_nodes,\
                                       time_dependent=self.time_dependent,num_layers=self.num_layers, param_inits=inits[i], init = self.initializer,\
                                       nn_type = self.nn_type, activation=self.activation, dtype=dtype) for i in range(self.num_components)]
        # build the ARGMMBlocks
        for i, block in enumerate(self.armmbs):
            block(*self.domain_sampler(domain=[[0., 1.] for j in range(block.dim)], num_samples=1))
            if self.time_dependent:
                self.build_nn(obj=self, nn_name='c_' + str(i), input_dim=1, output_dim=1,\
                              num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                              b_f_initializer='ones')
            else:
                setattr(self, name, self.add_weight(name='c_' + str(i), shape=(), initializer="random_normal", trainable=True))


    def call(self, *args):
        probs = []
        coeffs = []
        for i, block in enumerate(self.armmbs):
            if self.time_dependent:
                c = self.call_nn(obj=self, nn_name='c_' + str(i), input=args[0],\
                                 num_layers=self.num_layers, activation=self.activation)
            else:
                c = getattr(self, 'c_' + str(i))
            probs.append(block(*args))
            coeffs.append(c)
        probs = tf.convert_to_tensor(probs, dtype=self.dtype)
        coeffs =  tf.nn.softmax(tf.convert_to_tensor(coeffs, dtype=self.dtype), axis=0)
        #print('probs======================>', probs)
        #print('coeffs=====================>', coeffs)
        #print('dot=========================>', probs * coeffs)
        return tf.math.reduce_sum(coeffs * probs, axis=0)
