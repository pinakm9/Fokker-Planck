import tensorflow as tf
import numpy as np

class LayerA(tf.keras.layers.Layer):
    def __init__(self):
        super(LayerA, self).__init__()

    def build(self, input_shape):
        self.a = self.add_weight(shape=(1,), initializer="ones", trainable=True)

    def call(self, input):
        return self.a * input

class LayerB(tf.keras.layers.Layer):
    def __init__(self):
        super(LayerB, self).__init__()

    def build(self, input_shape):
        self.b = self.add_weight(shape=(1,), initializer="ones", trainable=True)

    def call(self, input):
        return input + self.b


class LayerSq(tf.keras.layers.Layer):
    def __init__(self):
        super(LayerSq, self).__init__()

    def call(self, input):
        return input*input

class DiffLayer(tf.keras.layers.Layer):
    def __init__(self, func):
        super(DiffLayer, self).__init__()
        self.func = func

    def call(self, input):
        with tf.GradientTape() as tape:
            tape.watch(input)
            return tape.gradient(self.func(input), input)

class QuadModel(tf.keras.models.Model):
    def __init__(self):
        super(QuadModel, self).__init__()
        self.layerA = LayerA()
        self.layerB = LayerB()
        self.layerSq = LayerSq()

    def call(self, input):
        y = self.layerA(input)
        y = self.layerB(y)
        y = self.layerSq(y)
        return y


x = tf.constant([[1, 2], [3, 4]], dtype = tf.float32)
qmodel = QuadModel()
qmodel(x)
qmodel.summary()
print(tf.reduce_mean(tf.square(qmodel(tf.zeros_like(x))-49.)))

dqmodel = DiffLayer(qmodel)
d2qmodel = DiffLayer(dqmodel)


def loss(input, a = 5., b = 7.):
    loss_0 = tf.reduce_mean(tf.square(d2qmodel(input) - 2*a**2))
    loss_1 = tf.square(dqmodel(tf.constant(0., dtype=tf.float32)) - 2*a*b)
    loss_2 = tf.square(qmodel(tf.constant(0., dtype=tf.float32))  - b**2)
    return  (loss_0 + loss_1 + loss_2)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=1000, decay_rate=0.96, staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
for i in range(10000):
    x = tf.convert_to_tensor(np.random.rand(100, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss1 = loss(x)
    grads = tape.gradient(loss1, qmodel.trainable_weights)
    optimizer.apply_gradients(zip(grads, qmodel.trainable_weights))
        #print(grads)
        #print(qmodel.trainable_weights)
    print(loss1)
    if loss1 < 1e-16:
        break

print(qmodel.trainable_variables)
