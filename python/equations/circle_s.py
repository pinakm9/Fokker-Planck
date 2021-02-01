# add required folders to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
image_dir = str(script_dir.parent.parent)
print(image_dir)
sys.path.insert(0, module_dir + '/modules')
# import modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nnsolve as nn
import nnplot as plot
#tf.get_logger().setLevel('ERROR')
# set parameters of OU equation
d = 1.
space_x = [-3., 3.]
space_y = [-3., 3.]
infinity = 5.0
# create solver for OU equation
p = nn.ARGMM_F(dim = 2, num_components = 10, num_nodes = 2, num_LSTM_layers = 1, time_dependent = False, name='circle_s_F_solver')
p.add_domain([space_x, space_y])
p(*p.domain_sampler(p.domains[0], 1))
p.summary()

class DiffOp(tf.keras.layers.Layer):
    def __init__(self, func):
        super(DiffOp, self).__init__()
        self.func = func

    def call(self, x, y):
        x2 = x*x
        y2 = y*y
        z = -4.0*(x2 + y2 - 1.0)
        with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
            outer_x.watch(x)
            outer_y.watch(y)
            with tf.GradientTape() as inner_x, tf.GradientTape() as inner_y:
                inner_x.watch(x)
                inner_y.watch(y)
                p_ = self.func(x, y)
                p_x = inner_x.gradient(p_, x)
                p_y = inner_y.gradient(p_, y)
            p_xx = outer_x.gradient(p_x, x)
            p_yy = outer_y.gradient(p_y, y)
        a = (x*z + y) * p_x #(-12.0*x2 - 4.0*y2 + 4.0)*p_
        b = (y*z - x) * p_y #(-12.0*y2 - 4.0*x2 + 4.0)*p_
        c = (4*z - 8.0)*p_
        return a + b + c - 0.5*(p_xx + p_yy)

df_op = DiffOp(func=p)
p.add_objective(df_op)
#p.add_domain([space_x, space_y])
#p.add_objective(lambda y: p(infinity * tf.ones_like(y), y))# - 0.*tf.exp(-0.5*tf.square(x))/tf.sqrt(2.*np.pi))
#p.add_domain([space_y])
#p.add_objective(lambda y: p(-infinity * tf.ones_like(y), y))# - 0.*tf.exp(-0.5*tf.square(x))/tf.sqrt(2.*np.pi))
#p.add_domain([space_y])
#p.add_objective(lambda x: p(x, infinity * tf.ones_like(x)))# - 0.*tf.exp(-0.5*tf.square(x))/tf.sqrt(2.*np.pi))
#p.add_domain([space_x])
#p.add_objective(lambda x: p(x, -infinity * tf.ones_like(x)))# - 0.*tf.exp(-0.5*tf.square(x))/tf.sqrt(2.*np.pi))
#p.add_domain([space_x])
#p.add_objective(lambda *args: 1.0 - p(tf.zeros_like(args[0]), tf.zeros_like(args[0]), tf.zeros_like(args[0])))
#p.add_domain([space_x])

# learn the solution
p.solve_static(num_steps = 1000, num_samples = [1000, 50, 50, 50, 50], initial_rate = 0.01, decay_steps=500, decay_rate=0.1)
p.summary()
plot.plot_nn_2d_s(nn_sol=p, space=[space_x, space_y],\
           x_lim = space_x, y_lim = space_y, z_lim = [0., 0.5], folder=image_dir + '/images')
