import tensorflow as tf
import numpy as np

class x2y3Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(x2y3Layer, self).__init__()

    def call(self, *args):
        x, y = args
        return x**2*y**3


class DiffLayer(tf.keras.layers.Layer):
    def __init__(self, func):
        super(DiffLayer, self).__init__()
        self.func = func

    @tf.function
    def call(self, x, y):
        z = self.func(x, y)
        return tf.gradients(z, x)[0]

class Diff2Layer(tf.keras.layers.Layer):
    def __init__(self, func):
        super(Diff2Layer, self).__init__()
        self.func = func

    @tf.function
    def call(self, x, y):
        z = self.func(x, y)
        return tf.gradients(z, y)[0]


class Partial(tf.keras.layers.Layer):
    def __init__(self, func, i):
        super(Partial, self).__init__()
        self.func = func
        self.i = i

    def call(self, *args):
        with tf.GradientTape() as tape:
            tape.watch(args[self.i])
            y = self.func(*args)
            return tape.gradient(y, args[self.i])


l = x2y3Layer()
dl = DiffLayer(l)
d2l = DiffLayer(dl)
dyl = Diff2Layer(l)
x = tf.constant([1, 3], dtype = tf.float32)
y = tf.constant([2, 4], dtype = tf.float32)
print(l(x, y))
print(dl(x, y))
print(d2l(x, y))
print(dyl(x, y))

# Starting Partial test
l_x = Partial(l, 0)
l_xx = Partial(l_x, 0)
print(l_x(x, y))
print(l_xx(x, y))
