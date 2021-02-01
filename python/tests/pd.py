import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

class x2y3z(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(dtype=tf.float64)

    def call(self, x, y, z):
        return x**2 * y**3 * z


def gen_arg(num):
    return [tf.random.uniform(shape=(num, 1), minval=1.0, maxval=10.0, dtype=tf.float64) for d in range(3)]

f = x2y3z()
x, y, z = gen_arg(10)
print(x, y, z)
print(f(x, y, z))

def diff_op(x, y, z):
    with tf.GradientTape() as outer:
        outer.watch([x])
        with tf.GradientTape() as inner:
            inner.watch([x, y, z])
            f_ = f(x, y, z)
        grad_f = inner.gradient(f_, [x, y, z])
        f_x = grad_f[0]
        f_y = grad_f[1]
        f_z = grad_f[2]
    f_xx = outer.gradient(f_x, x)
    return f_y + f_z - f_xx

print(diff_op(x, y, z) - (x**2 * y**3 + 3 * x**2 * y**2 * z - 2 * y**3 * z))
