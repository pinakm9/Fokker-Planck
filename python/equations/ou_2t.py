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
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import nnsolve2 as nn
import nnplot2 as plot
import utility as ut
tf.get_logger().setLevel('ERROR')
# set parameters of OU equation
d = 1.
beta = 2.
b = tf.constant([[1., beta,  beta]], dtype=tf.float32)
space_x = [-5., 5.]
space_y = [-5., 5.]
t = [0., 100.]
# create solver for OU equation
ou = nn.ARGMM(dim = 3, num_components = 1, num_nodes = 1, num_LSTM_layers = 10, time_dependent = True)

"""
ou_t = nn.Partial(ou, 0)
ou_x  = nn.Partial(ou, 1)
ou_y = nn.Partial(ou, 2)
ou_xx = nn.Partial(ou_x, 1)
ou_yy = nn.Partial(ou_y, 2)
xou = lambda input: input[:, 1] * ou(input)
you = lambda input: input[:, 2] * ou(input)
xou_x = nn.Partial(xou, 1)
you_y = nn.Partial(you, 2)
"""
xou = lambda input: input[:, 1] * ou(input)
you = lambda input: input[:, 2] * ou(input)
# loss functions and domains
@ut.timer
def diff_op(input):
    vector0 =  tf.constant([[1., 0., 0.]])
    vector1 =  tf.constant([[0., 1., 0.]])
    vector2 =  tf.constant([[0., 0., 1.]])
    ou_t = nn.jvp(func=ou, input=input, vector=tf.concat([vector0 for i in range(tf.shape(input)[0])], axis=0))
    xou_x = nn.jvp(func=xou, input=input, vector=tf.concat([beta*vector1 for i in range(tf.shape(input)[0])], axis=0))
    you_y = nn.jvp(func=xou, input=input, vector=tf.concat([beta*vector2 for i in range(tf.shape(input)[0])], axis=0))
    ou_xx = nn.hvp(func=ou, input=input, vector=tf.concat([d*vector1 for i in range(tf.shape(input)[0])], axis=0))[:, 1]
    ou_yy = nn.hvp(func=ou, input=input, vector=tf.concat([d*vector2 for i in range(tf.shape(input)[0])], axis=0))[:, 2]
    #print(xou_x, you_y)
    return xou_x + you_y - ou_t + ou_xx + ou_yy

"""
def diff_op_old(input):
    ou_t(input) - beta*xou_x(input) - beta*you_y(input)- d*ou_xx(input) - d*ou_yy(input)
"""

ou.add_objective(diff_op)
ou.add_domain([t, space_x, space_y])
ou.add_objective(ou)# - 0.*tf.exp(-0.5*tf.square(x))/tf.sqrt(2.*np.pi))
ou.add_domain([(0.0, 0.0), space_x, space_y])
#ou.add_objective(lambda *args: 1.0 - ou(tf.zeros_like(args[0]), tf.zeros_like(args[0]), tf.zeros_like(args[0])))
#ou.add_domain([space_x])
#"""


print(ou(ou.domains[0].sample(1)))
input = ou.domains[0].sample(2)
vector = ou.domains[0].sample(2)
print(nn.hvp(ou, input, input))
print(nn.jvp(ou, input, vector))
print(ou.objectives[0](ou.domains[0].sample(10)))
print(ou(input), input[:, 1])
print(xou(input))
print(diff_op(input))
ou.summary()

def steady_sol(t, x, y):
    X = np.array([x, y])
    return np.exp(-np.dot(X, X))/(np.sqrt(2.) * np.pi)

ou.solve(num_steps = 500, num_samples = [1000, 100, 1], initial_rate = 0.001)
plot.error_heatmap(nn_sol=ou, true_sol=steady_sol, space=[space_x, space_y], t=100., file_path=image_dir + '/images/ou_2d_aeh.png')
