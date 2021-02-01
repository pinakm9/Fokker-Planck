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
alpha = 1.
beta = 4.
space = [-10., 10.]
t = [0., 10.]
# create solver for OU equation
ou = nn.ARGMM(dim = 2, num_components = 1, num_nodes = 1, num_LSTM_layers = 1, time_dependent = True)
ou_t = nn.Partial(ou, 0)
ou_x  = nn.Partial(ou, 1)
ou_xx = nn.Partial(ou_x, 1)
xou = lambda t, x: x * ou(t, x)
xou_x = nn.Partial(xou, 1)

# loss functions and domains
ou.add_objective(lambda t, x: ou_t(t, x) - alpha*xou_x(t, x) - beta*ou_xx(t, x))
ou.add_domain([t, space])
ou.add_objective(lambda x: ou(tf.zeros_like(x), x))# - 0.*tf.exp(-0.5*tf.square(x))/tf.sqrt(2.*np.pi))
ou.add_domain([space])
#ou.add_objective(lambda *args: 1.0 - ou(tf.zeros_like(args[0]), tf.zeros_like(args[0])))
#ou.add_domain([space])


def true_sol(t, x):
    b = 2.*beta*(1-np.exp(-2*alpha*t))
    return np.exp(-alpha*x**2/b) * np.sqrt(alpha/(np.pi*b))

def steady_sol(x):
    return np.sqrt(alpha/(2.*np.pi*beta)) * np.exp(-0.5*alpha*x**2/beta)

# learn the solution
ou.solve(num_steps = 500, num_samples = [10000, 100, 1], initial_rate = 0.001)
ou.summary()
plot.nn_vs_true_1d(nn_sol=ou, true_sol=true_sol, space=[[-15., 15.]], t=[0.1, 10.], x_lim = [-15., 15.], y_lim = [0., 0.5],\
                   folder=image_dir + '/images', video_name='ou_1d')
