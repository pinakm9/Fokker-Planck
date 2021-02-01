import tensorflow as tf
import nnsolver as nns
import build_layers as bl
import density as ds
import numpy as np

def get_param_inits(domain):
    mu = [tf.keras.initializers.RandomUniform(minval=d[0], maxval=d[1]) for d in domain]
    sigma = [tf.keras.initializers.Ones() for d in domain]
    return [mu, sigma]

def standard_normal(*args):
    sum_sq = tf.reduce_sum(tf.concat([arg**2 for arg in args], axis=1), axis=1)
    return tf.exp(-0.5*sum_sq)/(2.50662827463)**(len(args))


class SumOfProductsT(nns.NNSolver):
    def __init__(self, dim, principal_domain, num_components=5, num_nodes=50, num_layers=1,\
                 init='random_normal', nn_type='LSTM', activation=tf.keras.activations.tanh, name='SumOfProductsT_Model', dtype=tf.float32):
        super(SumOfProductsT, self).__init__(dim=dim, name=name, dtype=dtype)
        self.num_components = num_components
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.activation = activation
        self.nn_type = nn_type
        self.build_nn = getattr(bl, 'build_' + self.nn_type)
        self.call_nn = getattr(bl, 'call_' + self.nn_type)
        self.initializer = init
        self.sqrt_2pi = 2.50662827463
        self.principal_domain = principal_domain
        self.build_model()


    def build_model(self):
        """
        Builds 2 x space_dimension networks to account for all the components in the model
        """
        for d in range(self.dim - 1):
            self.build_nn(obj=self, nn_name='alpha_' + str(d), input_dim=d + 1, output_dim=self.num_components,\
                          num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                          b_f_initializer='zeros')
            self.build_nn(obj=self, nn_name='beta_' + str(d), input_dim=d + 1, output_dim=self.num_components,\
                          num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                          b_f_initializer='zeros')
        self.build_nn(obj=self, nn_name='c', input_dim=1, output_dim=self.num_components,\
                      num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer)

    def call(self, *args):
        prob = 1.
        # compute the parameters
        for d in range(self.dim - 1):
            alpha = self.call_nn(obj=self, nn_name='alpha_' + str(d), input=tf.concat(args[: d + 1], 1),\
                              num_layers=self.num_layers, activation=self.activation)
            beta = self.call_nn(obj=self, nn_name='beta_' + str(d), input=tf.concat(args[: d + 1], 1),\
                                 num_layers=self.num_layers, activation=self.activation)
            alpha = tf.keras.activations.softplus(alpha)
            beta = tf.keras.activations.softplus(beta)
            a, b = self.principal_domain[d]
            print('prduct=========>', (args[d+1] - a)**(alpha - 1.0) * (b - args[d+1])**(beta - 1.0)/ (b - a)**(alpha + beta - 1.0))
            normalizer = ((b - a)**(alpha + beta - 1.0)) * tf.exp(tf.math.lbeta(tf.stack([alpha, beta], axis=-1)))
            #print('alpha================>', alpha)
            #print('beta================>', beta)
            #print('normalizer================>', normalizer)
            #print('x ===================>', args[d+1])
            #print('(x-a)^(alpha-1)======>', ((args[d+1] - a)**(alpha - 1.0)) * ((b - args[d+1])**(beta - 1.0))/normalizer)
            prob *= ((args[d+1] - a)**(alpha - 1.0)) * ((b - args[d+1])**(beta - 1.0))/normalizer
            print('prob=================>', prob)
        c = self.call_nn(obj=self, nn_name='c', input=args[0], num_layers=self.num_layers, activation=self.activation)
        c = tf.keras.activations.softmax(c, axis=1)
        return tf.math.reduce_sum(c * prob, axis=1)
