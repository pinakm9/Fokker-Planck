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

def true_grads(x, w, a=5., b=0.):
    a_, b_ = w
    x_ = x.numpy()[0]
    #a_ = a__.numpy()
    #b_ = b__.numpy()
    A = (2*a_**2*x_ + 2*a_*b_ - 2*a**2*x_ - 2*a*b)
    A2 = A*A
    del_a = 2*A*(4*a_*x_ + 2*b_)
    del_b = 2*A*(2*a_) + 4*b_*(b_-b)
    return [del_b, del_a]


def loss(model, input, a = 5., b = 0.):
    with tf.GradientTape() as tape:
        tape.watch(input)
        y = model(input)
    dy_dx = tape.gradient(y, input)
    loss_1 = tf.reduce_mean(tf.square(dy_dx - 2*a*tf.sqrt(y)))
    loss_2 = tf.square(model.layerB(tf.constant(0., dtype=tf.float32))  - b)[0]
    return  (loss_1 + loss_2)

x = tf.constant([[1, 2], [3, 4]], dtype = tf.float32)
qmodel = QuadModel()
qmodel(x)
qmodel.summary()
print(tf.reduce_mean(tf.square(qmodel(tf.zeros_like(x))-49.)))
print(loss(qmodel, x))


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=1000, decay_rate=0.96, staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
for i in range(10000):
    x = tf.constant(np.random.rand(100, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss1 = loss(qmodel, x)
    grads = tape.gradient(loss1, qmodel.trainable_weights)
    optimizer.apply_gradients(zip(grads, qmodel.trainable_weights))
        #print(grads)
        #print(qmodel.trainable_weights)
    print(loss1)
    if loss1 < 1e-16:
        break

print(qmodel.trainable_variables)
