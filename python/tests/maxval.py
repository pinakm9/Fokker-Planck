import tensorflow as tf
import numpy as np

def Z(D):
    return 0.5 * tf.sqrt(np.pi**3 * D) * (1.0 + tf.math.erf(1.0/tf.sqrt(D)))

print(1.0/Z(0.1))
