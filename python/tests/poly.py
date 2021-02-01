import tensorflow as tf

r = tf.constant([0., 1., 2., 3.])
r = tf.reshape(r, (-1, 1))
r_ = tf.constant([0., -1., 2., 3.])
r_ = tf.reshape(r_, (-1, 1))
R = tf.concat([r**n for n in range(3)], axis=1)
R_ = tf.concat([r_**n for n in range(3)], axis=1)
x = tf.concat([R, r, r_], axis=1)
print(x)
