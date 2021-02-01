import tensorflow as tf
import numpy as np
import pandas as pd
import json



with open('../data/config.json') as config_file:
    config = json.load(config_file)
dtype = getattr(tf, config['dtype'])
D = config['diffusion']
R = config['radius']
log_2_pi = tf.cast(tf.math.log(2.0*np.pi), dtype=dtype)
_2_pi = tf.cast(2.0*np.pi, dtype=dtype)

def diff_op(p, r):
    r2 = r*r
    z = 4.0*(r2 - 1.0)
    with tf.GradientTape() as outer_r:
        outer_r.watch(r)
        with tf.GradientTape() as inner:
            inner.watch(r)
            p_ = p(r)
            #print(p_)
        p_r = inner.gradient(p_, r)
    p_rr = outer_r.gradient(p_r, r)
    a = (D + z*r2) * p_r
    b = 4.0 * r * (z + 2.0)
    c = D * r * (p_rr - p_r**2)
    eqn = a - b + c
    return tf.reduce_mean(eqn**2)

def bdry_cond(p):
    return tf.reduce_mean(tf.exp(-p(tf.constant([[R]], dtype=dtype)))**2)
