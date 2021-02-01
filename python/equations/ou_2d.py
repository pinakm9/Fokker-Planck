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
beta = 2.
space_x = [-5., 5.]
space_y = [-5., 5.]
t = [0., 10.]
# create solver for OU equation
ou = nn.ARGMM(dim = 3, num_components = 10, num_nodes = 1, num_layers = 1, time_dependent = True)
ou_t = nn.Partial(ou, 0)
ou_x  = nn.Partial(ou, 1)
ou_y = nn.Partial(ou, 2)
ou_xx = nn.Partial(ou_x, 1)
ou_yy = nn.Partial(ou_y, 2)
xou = lambda t, x, y: x * ou(t, x, y)
you = lambda t, x, y: y * ou(t, x, y)
xou_x = nn.Partial(xou, 1)
you_y = nn.Partial(you, 2)
# loss functions and domains
ou.add_objective(lambda t, x, y: ou_t(t, x, y) - beta*xou_x(t, x, y) - beta*you_y(t, x, y)- d*ou_xx(t, x, y) - d*ou_yy(t, x, y))
ou.add_domain([t, space_x, space_y])
ou.add_objective(lambda x, y: ou(tf.zeros_like(x), x, y))# - 0.*tf.exp(-0.5*tf.square(x))/tf.sqrt(2.*np.pi))
ou.add_domain([space_x, space_y])
#ou.add_objective(lambda *args: 1.0 - ou(tf.zeros_like(args[0]), tf.zeros_like(args[0]), tf.zeros_like(args[0])))
#ou.add_domain([space_x])


def steady_sol(t, x, y):
    X = np.array([x, y])
    return np.exp(-np.dot(X, X))/(np.sqrt(2.) * np.pi)

# learn the solution
ou.solve(num_steps = 500, num_samples = [1000, 100, 1], initial_rate = 0.001)
ou.summary()
#plot.nn_2d(nn_sol=ou, true_sol=steady_sol, space=[space_x, space_y], t=[0.1, 10.],\
#           x_lim = space_x, y_lim = space_y, z_lim = [0., 0.5], folder=image_dir + '/images', video_name='ou_2d')
plot.error_heatmap(nn_sol=ou, true_sol=steady_sol, space=[space_x, space_y], t=10., file_path=image_dir + '/images/ou_2d_aeh.png')
