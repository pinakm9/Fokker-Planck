import tensorflow as tf
import numpy as np

class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def call(self, input):
        return input

class LayerA(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(LayerA, self).__init__()
        self.layer = layer

    def build(self, input_shape):
        self.a = self.add_weight(shape=(1,), initializer="ones", trainable=True)

    def call(self, input):
        return self.a * self.layer(input)

class LayerB(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(LayerB, self).__init__()
        self.layer = layer

    def build(self, input_shape):
        self.b = self.add_weight(shape=(1,), initializer="ones", trainable=True)

    def call(self, input):
        return self.layer(input) + self.b


class LayerSq(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(LayerSq, self).__init__()
        self.layer = layer

    def call(self, input):
        return tf.math.pow(self.layer(input), 2)

class DiffOp(tf.keras.layers.Layer):
    def __init__(self, func, a = 5.):
        super(DiffOp, self).__init__()
        self.func = func
        self.a = a

    def call(self, input):
        with tf.GradientTape() as tape:
            tape.watch(input)
            y = self.func(input)
            dy_dx = tape.gradient(y, input)
        return tf.math.pow(dy_dx - 1.0, 2) + tf.math.pow(self.func(tf.zeros_like(input)) - self.a, 2)

class QuadModel(tf.keras.models.Model):
    def __init__(self):
        super(QuadModel, self).__init__()
        self.layer = IdentityLayer()
        self.layer = LayerB(self.layer)

    def call(self, input):
        y = self.layer(input)
        print(y)
        self.add_loss(DiffOp(self.layer)(input))
        return y

x = tf.constant([[1, 2], [3, 4]], dtype = tf.float64)
qmodel = QuadModel()
qmodel.compile(optimizer = 'adam')
print(qmodel(x))
print(qmodel.summary())
data_x = tf.constant(np.random.rand(1000, 1))
data_y = tf.constant(np.zeros((1000, 1)))
qmodel.fit(x = data_x, y = data_y, epochs = 380)
print(qmodel.trainable_variables)
