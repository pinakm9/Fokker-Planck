# Implentation of an equation using eqn module
import tensorflow as tf
import tensorflow_probability as tfp
import eqn
import solve


def diff_op(func, input_, a=0.3, b=0.5, sigma=0.1):
    input = tf.convert_to_tensor(input_)
    num_x = tf.shape(input)[0]
    dim = tf.shape(input)[1]
    with tf.GradientTape() as tape2:
        tape2.watch(input)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(input)
            fx = func(input)
            c = tf.reshape(a*input - b*tf.math.pow(input, 3), (num_x, dim))
            c_fx = tf.keras.layers.Multiply()([c, fx])
            dc_fx = tape1.gradient(c_fx, input)
            d_fx = tape1.gradient(fx, input)
        print("debug {}, {}".format(input, type(input)))
        d2_fx = tape2.gradient(d_fx, input)
    return  -dc_fx + 0.5 * sigma**2 * d2_fx

def init_cond(input):
    return tfp.distributions.Normal(loc=0., scale=3.).prob(input)

def bdry_cond(input):
    return 0.

#"""
pde = eqn.QuasiLinearPDE0(diff_op, init_cond, bdry_cond, [[-10., 10.]], [0., 10.])
test_input = tf.constant([[1.0], [4.0]], dtype=tf.float32)
model = solve.dgm_model(dim=1, num_nodes=50, num_hidden_layers=4)
print(pde.loss(model, test_input))
solver  = solve.DGMSolver(eq = pde, num_nodes = 50, num_hidden_layers = 3)
#solver.solve()
