# add required folders to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
image_dir = str(script_dir.parent.parent)
print(image_dir)
sys.path.insert(0, module_dir + '/modules')
# import modules
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import nnsolve2 as nn
import nnplot2 as plot
#tf.get_logger().setLevel('ERROR')

class x2y3(tf.keras.layers.Layer):
    def __init__(self):
        super(x2y3, self).__init__()

    def build(self, input_shape):
        self.a = self.add_weight(shape=(1,), initializer="ones", trainable=True)

    def call(self, input):
        return input[:, 0]**2 * input[:, 1]**3


x = tf.constant([[1., 3.], [-1., 2.]], dtype=tf.float32)
v = tf.constant([[0., 1.], [0., 1.]], dtype=tf.float32)
f = x2y3()
print(f(x))
print(nn.hvp(f, x, v)[:, 1])
print(nn.jvp(f, x, v))
