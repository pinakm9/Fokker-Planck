import tensorflow as tf
import numpy as np
import pandas as pd
import json

with open('../data/config.json') as config_file:
    config = json.load(config_file)
dtype = getattr(tf, config['dtype'])
quad_data = pd.read_csv(config['data_path'] + '/' + config['quadrature_file'])
weights =  tf.reshape(tf.constant(np.array(quad_data['weights']).reshape((-1, 1)), dtype=dtype), shape=(-1, 1))
abscissae = tf.reshape(tf.constant(np.array(quad_data['abscissae']).reshape((-1, 1)), dtype=dtype), shape=(-1, 1))
D = config['diffusion']
R = config['radius']
pts = 0.5 * R * (abscissae + 1.0)
sigma2 = config['initial_variance']
log_2_pi_sigma = tf.cast(tf.math.log(2.0*np.pi*np.sqrt(sigma2)), dtype=dtype)
_2_pi = tf.cast(2.0*np.pi, dtype=dtype)

def Gauss_quad(f, t):
    ones = tf.ones_like(pts, dtype=dtype)
    tensor = tf.convert_to_tensor([0.5 * R * tf.reduce_sum(weights * f(t_[0]*ones, pts) * pts) for t_ in t], dtype=dtype)
    return tf.reshape(tensor,  shape=(-1, 1))

def normalized(f, t, r):
    c = _2_pi * Gauss_quad(f, t)
    return f(t, r) / c

def diff_op(p, t, r):
    r2 = r*r
    z = 4.0*(r2 - 1.0)
    with tf.GradientTape() as outer_r:
        outer_r.watch(r)
        with tf.GradientTape() as inner:
            inner.watch([t, r])
            p_ = normalized(p, t, r)
            #print(p_)
        grad_p = inner.gradient(p_, [t, r])
        p_t = grad_p[0]
        p_r = grad_p[1]
    p_rr = outer_r.gradient(p_r, r)
    a = r * z * p_r
    b = 4.0 * p_ * (z + 2.0)
    eqn = r*(a + b - p_t) + D * (r * p_rr + p_r)
    return tf.reduce_mean(eqn**2)

def init_cond(p, r):
    return tf.reduce_mean((normalized(p, tf.zeros_like(r, dtype=dtype), r) - tf.exp(-0.5*r*r - log_2_pi_sigma))**2)
