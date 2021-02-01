import tensorflow as tf

class Partial(tf.keras.layers.Layer):
    """
    Class for defining partial derivative of a function
    """
    def __init__(self, func, i):
        super(Partial, self).__init__()
        self.func = func
        self.i = i

    def call(self, *args):
        with tf.GradientTape() as tape:
            tape.watch(args[self.i])
            y = self.func(*args)
            return tape.gradient(y, args[self.i])


def hvp(func, vector, *args):
    """
    Computes Hessian vector products
    """
    with tf.autodiff.ForwardAccumulator(input, vector) as acc:
        with tf.GradientTape() as tape:
            y = func(*args)
    backward = tape.gradient(y, tf.concat(args, 1))
    return acc.jvp(backward)  # forward-over-backward Hessian-vector product
