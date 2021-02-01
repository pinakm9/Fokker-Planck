import tensorflow as tf
import numpy as np
from modules import utility as ut

def domain_sampler(domain, num_samples, dtype = tf.float32):
    return [tf.random.uniform(shape=(num_samples, 1), minval=d[0], maxval=d[1], dtype=dtype) for d in domain]


@ut.timer
def trapezoidal_1d(nn, domain, time=None, num_pts=1000):
    x = tf.reshape(tf.convert_to_tensor(np.linspace(domain[0], domain[1], num=num_pts), dtype=nn.dtype), (num_pts, 1))
    if time is None:
     y = nn(x)
    else:
     y = nn(time * tf.ones_like(x), x)
    return (2.0*tf.reduce_sum(y) - tf.reduce_sum(y[0] + y[-1])) * (domain[1] - domain[0])/(2.0*(num_pts - 1))

def trapezoidal_2d(nn, domain, time=None, num_pts=100):
    x = tf.reshape(tf.convert_to_tensor(np.linspace(domain[0][0], domain[0][1], num=num_pts), dtype=nn.dtype), (num_pts, 1))
    y = tf.reshape(tf.convert_to_tensor(np.linspace(domain[1][0], domain[1][0], num=num_pts), dtype=nn.dtype), (num_pts, 1))
    dx, dy = (domain[0][1] - domain[0][0])/(num_pts - 1), (domain[1][1] - domain[1][0])/(num_pts - 1)
    if time is None:
        nn_ = nn
    else:
        def nn_(x, y):
            t = time * tf.ones_like(x)
            return nn(t, x, y)
    term_1 = dx*dy*tf.reduce_sum(tf.convert_to_tensor([tf.reduce_sum(nn_(x[1:-1], y[j] * tf.ones_like(x[1:-1]))) for j in range(1, num_pts-1)]))
    term_2 = 0.5*dx*dy*tf.reduce_sum(nn_(x[1:-1], y[0][0] * tf.ones_like(x[1:-1])))
    term_3 = 0.5*dx*dy*tf.reduce_sum(nn_(x[1:-1], y[-1][0] * tf.ones_like(x[1:-1])))
    term_4 = 0.5*dx*dy*tf.reduce_sum(nn_(x[0][0] * tf.ones_like(y[1:-1]), y[1:-1]))
    term_5 = 0.5*dx*dy*tf.reduce_sum(nn_(x[-1][0] * tf.ones_like(y[1:-1]), y[1:-1]))
    term_6 = 0.25*dx*dy*(tf.reduce_sum(nn_(x[:1], y[:1]) + nn_(x[-1:], y[:1]) + nn_(x[-1:], y[-1:]) + nn_(x[:1], y[-1:])))
    return (term_1 + term_2 + term_3 + term_4 + term_5 + term_6)

def monte_carlo(nn, domain, time=None, num_pts=1000):
    if time is None:
        nn_ = nn
    else:
        def nn_(*args):
            t = time * tf.ones_like(args[0])
            return nn(t, *args)
    args = domain_sampler(domain, num_pts, dtype=nn.dtype)
    V = np.prod([(d[1]-d[0]) for d in domain])
    return tf.reduce_sum(nn_(*args)) * (V/num_pts)
