import tensorflow as tf
import numpy as np
import pandas as pd
import json



with open('../data/config.json') as config_file:
    config = json.load(config_file)
dtype = getattr(tf, config['dtype'])
D = config['diffusion']
R = config['radius']
sigma2 = config['initial_variance']
log_2_pi_sigma = tf.cast(tf.math.log(2.0*np.pi*np.sqrt(sigma2)), dtype=dtype)
_2_pi = tf.cast(2.0*np.pi, dtype=dtype)

def mc_quad(f, t, r):
    ones = tf.ones_like(r, dtype=dtype)
    tensor = tf.convert_to_tensor([R * tf.reduce_mean(f(t_[0]*ones, r) * r) for t_ in t], dtype=dtype)
    return tf.reshape(tensor,  shape=(-1, 1))

def diff_op(p, t, r):
    r2 = r*r
    z = 4.0*(r2 - 1.0)
    with tf.GradientTape() as outer_r:
        outer_r.watch(r)
        with tf.GradientTape() as inner:
            inner.watch([t, r])
            p_ = p(t, r)
        grad_p = inner.gradient(p_, [t, r])
        p_t = grad_p[0]
        p_r = grad_p[1]
    p_rr = outer_r.gradient(p_r, r)
    a = r * z * p_r
    b = 4.0 * p_ * (z + 2.0)
    eqn = r*(a + b - p_t) + D * (r * p_rr + p_r)
    return tf.reduce_mean(eqn**2)

def init_cond(p, r):
    return tf.reduce_mean((p(tf.zeros_like(r, dtype=dtype), r) - tf.exp(-0.5*r*r/sigma2 - log_2_pi_sigma))**2)

def integral_cond(p, t, r):
    return tf.reduce_mean((_2_pi * mc_quad(p, t, r) - tf.ones_like(t, dtype=dtype))**2)
