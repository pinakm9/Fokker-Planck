# A module for computing derivatives of neural networks

import tensorflow as tf
import numpy as np
import solve
import utility as ut
from eqn1 import diff_op, test_loss

def grad(func, input, dx = 1e-6):
    """
    Description: Function to compute gradient

    Args:
        func: function to be differentiated
        input: tensor containing points at which the function is to be differentiated, shape = (None, dimension of domain of the function,)
        dx: step-size computing derivative, default = 1e-6

    Return:
        A tensor containing the computed partials, shape = (None, dim)
    """
    dim = tf.shape(input)[1]
    gradients = []
    for t in input:
        x = tf.reshape(t, (1, dim))
        partials = []
        for i in range(dim):
            h = np.zeros(dim)
            h[i] = dx
            h = tf.constant(h, dtype = tf.float64)
            partials.append((func(x + h) - func(x - h))/(2*dx))
        gradients.append(tf.concat(partials, 0))
    return tf.stack(gradients)


def hess(func, input, dx = 1e-4, dy = 1e-4):
    """
    Description: Function to compute gradient

    Args:
        func: function to be differentiated
        input: tensor containing points at which the function is to be differentiated, shape = (None, dimension of domain of the function,)
        dx: step-size computing first derivative, default = 1e-4
        dy: step-size computing second derivative, default = 1e-4

    Return:
        A tensor containing the computed partials, shape = (None, dim, dim)
    """
    dim = tf.shape(input)[1]
    hessians = []
    for t in input:
        x = tf.reshape(t, (1, dim))
        hessian = []
        for i in range(dim):
            hx = np.zeros(dim)
            hx[i] = dx
            hx = tf.constant(hx, dtype = tf.float64)
            row_partials = []
            for j in range(dim):
                hy = np.zeros(dim)
                hy[j] = dy
                hy = tf.constant(hy, dtype = tf.float64)
                left = (func(x + hx + hy) - func(x - hx + hy))/(2*dx)
                right = (func(x + hx - hy) - func(x - hx - hy))/(2*dx)
                row_partials.append((left-right)/(2*dy))
            hessian.append(tf.concat(row_partials, 0))
        hessians.append(tf.stack(hessian))
    return tf.stack(hessians)


def partial(func, input, i, dx = 1e-4):
    """
    Description: Function to compute partial derivative w.r.t a a single variable

    Args:
        func: function to be differentiated
        input: tensor containing points at which the function is to be differentiated, shape = (None, dimension of domain of the function,)
        i: index of the variable w.r.t the function is to be differentiated
        dx: step-size computing first derivative, default = 1e-4

    Return:
        A tensor containing the computed partials, shape = (None, 1)
    """
    dim = tf.shape(input)[1]
    partials = []
    hx = np.zeros(dim)
    hx[i] = dx
    hx = tf.constant(hx, dtype = tf.float64)
    for t in input:
        x = tf.reshape(t, (1, dim))
        partials.append((-func(x + 2*hx) + 8*func(x + hx) - 8*func(x - hx) + func(x - 2*hx))/(12*dx))
    return tf.stack(partials)


def mixed_partial(func, input, i, j, dx = 1e-4, dy = 1e-4):
    """
    Description: Function to compute mixed partials in two variables

    Args:
        func: function to be differentiated
        input: tensor containing points at which the function is to be differentiated, shape = (None, dimension of domain of the function,)
        i: index of the first variable w.r.t the function is to be differentiated
        j: index of the second variable w.r.t the function is to be differentiated
        dx: step-size computing first derivative, default = 1e-4
        dy: step-size computing second derivative, default = 1e-4

    Return:
        A tensor containing the computed mixed partials, shape = (None, 1)
    """
    dim = tf.shape(input)[1]
    partials = []
    hx = np.zeros(dim)
    hx[i] = dx
    hx = tf.constant(hx, dtype = tf.float64)
    hy = np.zeros(dim)
    hy[j] = dy
    hy = tf.constant(hy, dtype = tf.float64)
    for t in input:
        x = tf.reshape(t, (1, dim))
        left = (func(x + hx + hy) - func(x - hx + hy))/(2*dx)
        right = (func(x + hx - hy) - func(x - hx - hy))/(2*dx)
        partials.append((left-right)/(2*dy))
    return tf.stack(partials)


@tf.function
def loss(x):
    f = lambda x_: tf.reduce_sum(tf.math.square(tf.math.square(x_)), axis = 0)
    return tf.vectorized_map(f, x)

@ut.tester
@tf.function
def comp_single_hess(f, x):
    return tf.hessians(f(x), x)

#@tf.function
def comp_hess(f, x):
    hess = []
    for v in x:
        hess.append(comp_single_hess(f, [v]))
    return hess

@ut.tester
@tf.function
def comp_grad(f, x):
    return tf.gradients(f(x), x)

@ut.tester
@tf.function
def comp_partial(f, x, i):
    return tf.gradients(tf.gradients(f(x), x), x)

@ut.tester
@tf.function
def model_hess(x):
    return tf.hessians(model(x), x)


model = solve.dgm_model(dim=1, num_nodes=50, num_hidden_layers=4)
inp =  tf.constant([[1.0], [4.0]], dtype=tf.float32)
print(test_loss(model, inp))
