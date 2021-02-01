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
        y = self.layer(input)
        return self.square(y)

    @tf.function
    def square(self, input):
        return tf.math.pow(input, 2)

class DiffOp(tf.keras.layers.Layer):
    def __init__(self, func, a = 5., b = 7.):
        super(DiffOp, self).__init__()
        self.func = func
        self.a = a
        self.b = b

    def call(self, input):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input)
            y = self.func(input)
            dy_dx = tape.gradient(y, input)
        return tf.math.pow(dy_dx - 2*self.a**2 * input -2*self.a*self.b, 2) + tf.math.pow(self.func(tf.zeros_like(input)) - self.b**2, 2)

class QuadModel(tf.keras.models.Model):
    def __init__(self):
        super(QuadModel, self).__init__()
        self.layer = IdentityLayer()
        self.layer = LayerA(self.layer)
        self.layer = LayerB(self.layer)
        self.layer = LayerSq(self.layer)
        self.objective = DiffOp(self.layer)

    def call(self, input):
        y = self.layer(input)
        l = self.objective(input)
        print('loss = {}'.format(l))
        self.add_loss(l)
        return y

def true_grads(x, w, a=5., b=7.):
    a_, b_ = w
    x_ = x.numpy()[0]
    #a_ = a__.numpy()
    #b_ = b__.numpy()
    A = (2*a_**2*x_ + 2*a_*b_ - 2*a**2*x_ - 2*a*b)
    A2 = A*A
    del_a = 2*A*(4*a_*x_ + 2*b_)
    del_b = 2*A*(2*a_) + 4*b_*(b_**2-b**2)
    return [del_b, del_a]


x = tf.constant([[1, 2], [3, 4]], dtype = tf.float32)
qmodel = QuadModel()

data_x = tf.constant(np.random.rand(5000, 1))
data_y = tf.constant(np.zeros((10000, 1)))
# Iterate over epochs.
loss_metric = tf.keras.metrics.Mean()
epochs = 1

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(data_x):
        with tf.GradientTape() as tape:
            reconstructed = qmodel(x_batch_train)
            # Compute reconstruction loss
            loss = 0
            loss += sum(qmodel.losses)  # Add KLD regularization loss

        grads = tape.gradient(loss, qmodel.trainable_weights)
        w = [t.numpy()[0] for t in qmodel.trainable_weights]
        g = [t.numpy()[0] for t in grads]
        tg = true_grads(x_batch_train, qmodel.trainable_variables)
        print([t.numpy()[0] for t in tg], g)
        #print(w)
        #print('x = {}, w = {}, grads = {}, loss = {}'.format(x_batch_train, w, g, qmodel.losses))
        #print('true_grads = {}'.format())
        optimizer.apply_gradients(zip(tg, qmodel.trainable_weights))
        #print(qmodel.trainable_variables)

        loss_metric(loss)

        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))



print(qmodel.trainable_variables)
