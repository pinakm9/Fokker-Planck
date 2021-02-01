import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, activation=tf.keras.activations.sigmoid):
        super(Linear, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True)

        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, input):
        #print("dsfgjhdaskljfghsdzlkjvhgasdlkfh asdjkvhaklsejfh klajsdhvckzjxdhvbk,jasxfdfhkjasdhvbk.jzxchv nklasdxfnhvbjklsdfhfbv {} {}".format(input.shape, self.w))
        return self.activation(tf.matmul(input, self.w) + self.b)


class SinModel(tf.keras.models.Model):
    def __init__(self, input_dim=1,  num_nodes=50, activation=tf.keras.activations.tanh):
        super(SinModel, self).__init__()
        self.num_nodes = num_nodes
        self.activation = activation
        self.dl_1 = Linear(self.num_nodes, self.activation)
        self.dl_2 = Linear(self.num_nodes, self.activation)
        self.dl_3 = Linear(1, self.activation)

    def call(self, input):
        y = self.dl_1(input)
        y = self.dl_2(y)
        y = self.dl_3(y)
        return y

    def plot(self, interval = [0.0, 2*np.pi]):
        x = np.linspace(interval[0], interval[1], 100)
        x_t = tf.constant(np.reshape(x, (100, 1)), dtype = tf.float32)
        y_t = self.call(x_t)
        plt.plot(x_t.numpy(), y_t.numpy())
        plt.plot(x, np.sin(x))
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()

class DiffLayer(tf.keras.layers.Layer):
    def __init__(self, func):
        super(DiffLayer, self).__init__()
        self.func = func

    def call(self, input):
        with tf.GradientTape() as tape:
            tape.watch(input)
            return tape.gradient(self.func(input), input)


x = tf.constant([[1], [3]], dtype = tf.float64)
smodel = SinModel()
print(smodel(x))
print(smodel.summary())
dsmodel = DiffLayer(smodel)
d2smodel = DiffLayer(smodel)


def loss(input):
    zero = tf.constant([[0.]], dtype=tf.float32)
    loss_0 = tf.reduce_mean(tf.square(d2smodel(input) + smodel(input)))
    loss_1 = tf.square(dsmodel(zero) - 1.0)[0][0]
    loss_2 = tf.square(smodel(zero))[0][0]
    return loss_0 #+ loss_1 + loss_2


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=1000, decay_rate=0.96, staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
for i in range(1000):
    x = tf.convert_to_tensor(2*np.pi*np.random.rand(100, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss1 = loss(x)
        grads = tape.gradient(loss1, smodel.trainable_weights)
        optimizer.apply_gradients(zip(grads, smodel.trainable_weights))
        #print(grads)
        #print(smodel.trainable_weights)
    print(loss1)
    if loss1 < 1e-16:
        break

print(smodel.trainable_variables)
smodel.plot()
