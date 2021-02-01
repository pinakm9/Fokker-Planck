import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import tensorflow as tf
import numpy as np
import spg_1 as sp
import nnplot as nnp
import integrate as quad


r = 5.0
space_x = r * np.array([-1., 1.])
space_y = r * np.array([-1., 1.])
time_= [0., 10.0]
dtype=tf.float64

class RadialInit(tf.keras.initializers.Initializer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, shape, dtype=dtype):
        self.theta = tf.convert_to_tensor(np.linspace(-np.pi, np.pi, num=shape[0], dtype=np.float64), dtype=dtype)
        if self.dim == 0:
            return tf.cos(self.theta)
        else:
            return tf.sin(self.theta)



param_inits = sp.get_param_inits(domain=[space_x, space_y])
#"""
mu = [RadialInit(0), RadialInit(1)]
sigma = [tf.keras.initializers.Constant(value=-1.0) for d in [0, 1]]
param_inits = [mu, sigma]
#"""
nn = sp.SumOfProductsT(num_components = 100, dim = 3, num_nodes=1, num_layers = 1, param_inits=param_inits, dtype=dtype)
nn.add_domain([time_, space_x, space_y])

"""
plotter = nnp.NNPlotter(funcs=[nn], space=[space_x, space_y], num_pts_per_dim=50)
plotter.plot(file_path='../../images/spg_1_test.png', t=0.0)
#"""
# integration test
num_pts = 5000
t, x, y = nn.domain_sampler([time_, [-4.0, 4.0], [-4.0, 4.0]], num_pts)
print('#################################################################')
print(quad.monte_carlo(nn, [[-10.0, 10.0], [-10.0, 10.0]], time=-9.0, num_pts=10000))


param_inits = sp.get_param_inits(domain=[space_x])
nn = sp.SumOfProductsT(num_components = 7, dim = 2, num_nodes=1, num_layers = 1, init='zeros', param_inits=param_inits, dtype=dtype)
nn.add_domain([time_, space_x])
#"""
plotter = nnp.NNPlotter(funcs=[nn], space=[space_x], num_pts_per_dim=50)
plotter.plot(file_path='../../images/spg_1_test_1d.png', t=0.0)
#"""
# integration test
num_pts = 5000
t, x = nn.domain_sampler([time_, [-4.0, 4.0]], num_pts)

print(quad.trapezoidal_1d(nn, [-40.0, 40.0], time=-7.0, num_pts=100000))
x = tf.random.uniform(shape=(10000, 1), minval=-15.0, maxval=15., dtype=dtype)
print((3.0/1000)*tf.reduce_sum(tf.exp(-0.5*(x-10.0)**2)/(2.50662827463)))



nn = sp.SumOfProductsT(num_components = 7, dim = 5, num_nodes=1, num_layers = 1, init='zeros', dtype=dtype)
print(quad.monte_carlo(nn, [[-3.0, 3.0]]*4, time=-9.0, num_pts=10000))


nn = sp.SumOfProductsTI(num_components = 7, dim = 5, num_nodes=1, num_layers = 1, init='zeros', dtype=dtype)
print(quad.monte_carlo(nn, [[-3.0, 3.0]]*4, time=2.0, num_pts=10000))
print(quad.monte_carlo(sp.standard_normal, [[-3.0, 3.0]]*4, time=None, num_pts=10000))
