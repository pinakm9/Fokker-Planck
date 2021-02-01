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
    return tf.exp(-0.5*sum_sq)/(2.0*np.pi)**(len(args)/2.0)

def standard_normal_2(x, y):
    return 0.5*tf.exp(-0.5*(x*x + y*y))/np.pi

class SumOfProductsT(nns.NNSolver):
    def __init__(self, dim, principal_domain, num_components=5, num_nodes=50, num_layers=1,\
                 param_inits=None, init='random_normal', regularizer=None, nn_type='LSTM', activation=tf.keras.activations.tanh,\
                 name='SumOfProductsT_Model', dtype=tf.float32):
        super(SumOfProductsT, self).__init__(dim=dim, name=name, dtype=dtype)
        self.num_components = num_components
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.activation = activation
        self.nn_type = nn_type
        self.build_nn = getattr(bl, 'build_' + self.nn_type)
        self.call_nn = getattr(bl, 'call_' + self.nn_type)
        self.initializer = init
        self.sqrt_2pi = np.sqrt(2.0*np.pi)
        self.sqrt_2 = np.sqrt(2.0)
        self.principal_domain = principal_domain
        self.regularizer = regularizer
        if param_inits is None:
            self.param_inits = [[tf.keras.initializers.RandomNormal() for d in range(self.dim - 1)], \
                                [tf.keras.initializers.Ones() for d in range(self.dim - 1)]]
        else:
            self.param_inits = param_inits
        self.build_model()


    def build_model(self):
        """
        Builds 2 x space_dimension networks to account for all the components in the model
        """
        for d in range(self.dim - 1):
            self.build_nn(obj=self, nn_name='mu_' + str(d), input_dim=d + 1, output_dim=self.num_components,\
                          num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                          b_f_initializer=self.param_inits[0][d], regularizer=self.regularizer)
            self.build_nn(obj=self, nn_name='sigma_' + str(d), input_dim=d + 1, output_dim=self.num_components,\
                          num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                          b_f_initializer=self.param_inits[1][d], regularizer=self.regularizer)
        self.build_nn(obj=self, nn_name='c', input_dim=1, output_dim=self.num_components,\
                      num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer, regularizer=self.regularizer)

    def call(self, *args):
        prob = 1.
        # compute the parameters
        for d in range(self.dim - 1):
            mu = self.call_nn(obj=self, nn_name='mu_' + str(d), input=tf.concat(args[: d + 1], 1),\
                              num_layers=self.num_layers, activation=self.activation)
            sigma = self.call_nn(obj=self, nn_name='sigma_' + str(d), input=tf.concat(args[: d + 1], 1),\
                                 num_layers=self.num_layers, activation=self.activation)
            sigma = tf.keras.activations.exponential(sigma) + 1e-6
            a, b = self.principal_domain[d]
            p, q = 1.0, 1e-6
            mu = 0.5*(b + a) + (0.5*(b - a))*tf.keras.activations.sigmoid(mu)
            #sigma = tf.keras.activations.softplus(sigma) + 1e-6 #0.5*(p + q) + (0.5*(p - q))*
            sigma_ = self.sqrt_2*sigma

            truncation_factor = 0.5 * (tf.math.erf((b - mu)/sigma_) - tf.math.erf((a - mu)/sigma_))
            """
            if args[0].shape[0] == 1:
                print('sigma_', sigma_)
                print('a-mu/sigma_', (a - mu)/sigma_)
                print('b-mu/sigma_', (b - mu)/sigma_)
                print('a -b', (a - mu)/sigma_ - (b - mu)/sigma_)
                print('truncation_factor', truncation_factor)
            #print('mu======================>', mu, mu.shape)
            #print('sigma======================>', sigma, sigma.shape)
            #print('args[d+1]======================>', args[d+1], args[d+1].shape)
            #print('mu - args[d+1]======================>', mu - args[d+1], (mu - args[d+1]).shape)
            """
            prob *= tf.exp(-0.5*((mu - args[d + 1])/sigma)**2)/(self.sqrt_2pi * sigma * truncation_factor)
        c = self.call_nn(obj=self, nn_name='c', input=args[0],\
                         num_layers=self.num_layers, activation=self.activation)
        c = tf.keras.activations.softmax(c, axis=1)
        return tf.reshape(tf.math.reduce_sum(c * prob, axis=1), shape=(args[0].shape[0], 1))




































































class SumOfProducts(nns.NNSolver):
    def __init__(self, dim, principal_domain, num_components=5, num_nodes=50, num_layers=1,\
                 param_inits=None, init='random_normal', regularizer=None, nn_type='LSTM', activation=tf.keras.activations.tanh,\
                 name='SumOfProducts_Model', dtype=tf.float32):
        super(SumOfProducts, self).__init__(dim=dim, name=name, dtype=dtype)
        self.num_components = num_components
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.activation = activation
        self.nn_type = nn_type
        self.build_nn = getattr(bl, 'build_' + self.nn_type)
        self.call_nn = getattr(bl, 'call_' + self.nn_type)
        self.initializer = init
        self.sqrt_2pi = np.sqrt(2.0*np.pi)
        self.sqrt_2 = np.sqrt(2.0)
        self.principal_domain = principal_domain
        self.regularizer = regularizer
        if param_inits is None:
            self.param_inits = [[tf.keras.initializers.RandomNormal() for d in range(self.dim)], \
                                [tf.keras.initializers.Ones() for d in range(self.dim)]]
        else:
            self.param_inits = param_inits
        self.build_model()


    def build_model(self):
        """
        Builds 2 x space_dimension networks to account for all the components in the model
        """
        for d in range(self.dim):
            self.build_nn(obj=self, nn_name='mu_' + str(d), input_dim=d, output_dim=self.num_components,\
                          num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                          b_f_initializer=self.param_inits[0][d], regularizer=self.regularizer)
            self.build_nn(obj=self, nn_name='sigma_' + str(d), input_dim=d, output_dim=self.num_components,\
                          num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                          b_f_initializer=self.param_inits[1][d], regularizer=self.regularizer)
        self.build_nn(obj=self, nn_name='c', input_dim=0, output_dim=self.num_components,\
                      num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer, regularizer=self.regularizer)

    def call(self, *args):
        prob = 1.
        # compute the parameters
        for d in range(self.dim):
            mu = self.call_nn(obj=self, nn_name='mu_' + str(d), input=tf.concat(args[: d], 1) if d > 0 else args[0],\
                              num_layers=self.num_layers, activation=self.activation, input_dim=d)
            sigma = self.call_nn(obj=self, nn_name='sigma_' + str(d), input=tf.concat(args[: d], 1) if d > 0 else args[0],\
                                 num_layers=self.num_layers, activation=self.activation, input_dim=d)
            sigma = tf.keras.activations.softplus(sigma) + 1e-6
            a, b = self.principal_domain[d]
            p, q = 1.0, 1e-6
            mu = 0.5*(b + a) + (0.5*(b - a))*tf.keras.activations.sigmoid(mu)
            #sigma = tf.keras.activations.softplus(sigma) + 1e-6 #0.5*(p + q) + (0.5*(p - q))*
            sigma_ = self.sqrt_2*sigma

            truncation_factor = 0.5 * (tf.math.erf((b - mu)/sigma_) - tf.math.erf((a - mu)/sigma_))
            """
            if args[0].shape[0] == 1:
                print('sigma_', sigma_)
                print('a-mu/sigma_', (a - mu)/sigma_)
                print('b-mu/sigma_', (b - mu)/sigma_)
                print('a -b', (a - mu)/sigma_ - (b - mu)/sigma_)
                print('truncation_factor', truncation_factor)
            #print('mu======================>', mu, mu.shape)
            #print('sigma======================>', sigma, sigma.shape)
            #print('args[d+1]======================>', args[d+1], args[d+1].shape)
            #print('mu - args[d+1]======================>', mu - args[d+1], (mu - args[d+1]).shape)
            """
            prob *= tf.exp(-0.5*((mu - args[d])/sigma)**2)/(self.sqrt_2pi * sigma * truncation_factor)
        c = self.call_nn(obj=self, nn_name='c', input=args[0],\
                         num_layers=self.num_layers, activation=self.activation, input_dim=0)
        c = tf.keras.activations.softmax(c, axis=1)
        return tf.reshape(tf.math.reduce_sum(c * prob, axis=1), shape=(args[0].shape[0], 1))
















class SumOfProductsTI(nns.NNSolver):
    def __init__(self, dim, principal_domain, init_cond=standard_normal_2, num_components=5, num_nodes=50, num_layers=1,\
                 param_inits=None, init='random_normal', regularizer=None, nn_type='LSTM', activation=tf.keras.activations.tanh,\
                 name='SumOfProductsT_Model', dtype=tf.float32):
        super(SumOfProductsTI, self).__init__(dim=dim, name=name, dtype=dtype)
        self.num_components = num_components
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.activation = activation
        self.nn_type = nn_type
        self.build_nn = getattr(bl, 'build_' + self.nn_type)
        self.call_nn = getattr(bl, 'call_' + self.nn_type)
        self.initializer = init
        self.sqrt_2pi = np.sqrt(2.0*np.pi)
        self.sqrt_2 = np.sqrt(2.0)
        self.principal_domain = principal_domain
        self.regularizer = regularizer
        if param_inits is None:
            self.param_inits = [[tf.keras.initializers.RandomNormal() for d in range(self.dim - 1)], \
                                [tf.keras.initializers.Ones() for d in range(self.dim - 1)]]
        else:
            self.param_inits = param_inits
        self.init_cond = init_cond
        self.build_model()


    def build_model(self):
        """
        Builds 2 x space_dimension networks to account for all the components in the model
        """
        for d in range(self.dim - 1):
            self.build_nn(obj=self, nn_name='mu_' + str(d), input_dim=d + 1, output_dim=self.num_components,\
                          num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                          b_f_initializer=self.param_inits[0][d], regularizer=self.regularizer)
            self.build_nn(obj=self, nn_name='sigma_' + str(d), input_dim=d + 1, output_dim=self.num_components,\
                          num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer,\
                          b_f_initializer=self.param_inits[1][d], regularizer=self.regularizer)
        self.build_nn(obj=self, nn_name='c', input_dim=1, output_dim=self.num_components,\
                      num_nodes=self.num_nodes, num_layers=self.num_layers, initializer=self.initializer, regularizer=self.regularizer)

    def call(self, *args):
        prob = 1.
        # compute the parameters
        for d in range(self.dim - 1):
            mu = self.call_nn(obj=self, nn_name='mu_' + str(d), input=tf.concat(args[: d + 1], 1),\
                              num_layers=self.num_layers, activation=self.activation)
            sigma = self.call_nn(obj=self, nn_name='sigma_' + str(d), input=tf.concat(args[: d + 1], 1),\
                                 num_layers=self.num_layers, activation=self.activation)
            sigma = tf.keras.activations.softplus(sigma) + 1e-6
            a, b = self.principal_domain[d]
            p, q = 1.0, 1e-6
            mu = 0.5*(b + a) + (0.5*(b - a))*tf.keras.activations.sigmoid(mu)
            #sigma = tf.keras.activations.softplus(sigma) + 1e-6 #0.5*(p + q) + (0.5*(p - q))*
            sigma_ = self.sqrt_2*sigma

            truncation_factor = 0.5 * (tf.math.erf((b - mu)/sigma_) - tf.math.erf((a - mu)/sigma_))
            """
            if args[0].shape[0] == 1:
                print('sigma_', sigma_)
                print('a-mu/sigma_', (a - mu)/sigma_)
                print('b-mu/sigma_', (b - mu)/sigma_)
                print('a -b', (a - mu)/sigma_ - (b - mu)/sigma_)
                print('truncation_factor', truncation_factor)
            #print('mu======================>', mu, mu.shape)
            #print('sigma======================>', sigma, sigma.shape)
            #print('args[d+1]======================>', args[d+1], args[d+1].shape)
            #print('mu - args[d+1]======================>', mu - args[d+1], (mu - args[d+1]).shape)
            """
            prob *= tf.exp(-0.5*((mu - args[d + 1])/sigma)**2)/(self.sqrt_2pi * sigma * truncation_factor)
        c = self.call_nn(obj=self, nn_name='c', input=args[0],\
                         num_layers=self.num_layers, activation=self.activation)
        c = tf.keras.activations.softmax(c, axis=1)
        t = args[0]
        s = tf.exp(-t)
        prob_1 = (1.0 - s) * tf.reshape(tf.math.reduce_sum(c * prob, axis=1), shape=(args[0].shape[0], 1))
        prob_2 = s * self.init_cond(*args[1:])
        return prob_1 + prob_2
