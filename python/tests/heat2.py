import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nnsolve as nn

h = nn.DGM(2)
h_x = nn.Partial(h, 0)
h_xx = nn.Partial(h_x, 0)
h_t = nn.Partial(h, 1)
L = np.pi

# interior
h.add_objective(lambda x0, t0: h_t(x0, t0) - h_xx(x0, t0))
h.add_domain([[0., L], [0., 1.]])

# initial condition
h.add_objective(lambda x1: h(x1, tf.zeros_like(x1)) - tf.sin(x1))
h.add_domain([[0., L]])

# left boundary condition
h.add_objective(lambda t1: h(tf.zeros_like(t1), t1))
h.add_domain([[0., 1.]])

# right boundary condition
h.add_objective(lambda t1: h(L*tf.ones_like(t1), t1))
h.add_domain([[0., 1.]])

def true_sol(x, t):
    return np.exp(-t)*np.sin(x)

def plot_true_vs_nn(t = 0.5):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111)
    x = np.linspace(0., L, 100)
    x_t = tf.constant(np.reshape(x, (100, 1)), dtype = tf.float32)
    ax.plot(x, true_sol(x, t))
    ax.plot(x, h(x_t, t*tf.ones_like(x_t)))
    plt.savefig('../../images/heat_sols.png')

def error_heatmap(s=50):
    err = np.zeros((s, s))
    for i in range(s):
        x = (L*i)/(s-1)
        x_t = tf.constant([[x]], dtype=tf.float32)
        for j in range(s):
            t = float(j)/(s-1)
            t_t = tf.constant([[t]], dtype=tf.float32)
            true = true_sol(x, t)
            nn = h(x_t, t_t).numpy()[0]
            err[i][j] = abs(true - nn)

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(err, cmap='viridis')
    fig.colorbar(im)
    plt.savefig('../../images/heat_error.png')

h.solve(num_steps = 500, num_samples = [1000, 100, 100, 100])

plot_true_vs_nn(t=0.5)
error_heatmap()
