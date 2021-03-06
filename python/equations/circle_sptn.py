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
import numpy as np
import sptn_1 as sp
import nnplot as nnp
import utility as ut
import integrate as quad

r = 2.0
space_x = r * np.array([-1., 1.])
space_y = r * np.array([-1., 1.])
time_ = [0.0, 10.0]


#num_components, param_inits = circle_params(25)
num_nodes = 10
num_layers = 3
nn_type = 'LSTM'
num_components = 10
model_name = 'circle_sptnI_1_{}_{}_{}_{}'.format(num_components, nn_type, num_nodes, num_layers)


p = sp.SumOfProductsT(dim = 3, principal_domain = [space_x, space_y], num_components=num_components,\
                       num_nodes=num_nodes, num_layers = num_layers, dtype=tf.float64, regularizer=tf.keras.regularizers.L2(l2=0.01),\
                       activation=tf.keras.activations.tanh)
p.add_domain([time_, space_x, space_y])
#"""
try:
    p.load_weights('saved models/' + model_name).expect_partial()
except:
    pass
#"""
print(p(*p.domain_sampler(0, 10)))
p.summary()
print('integral 1:', quad.monte_carlo(p, p.principal_domain, time=2.7))
one = tf.constant([[1.]], dtype=p.dtype)
#@ut.timer
def diff_op(t, x, y):
    z = -4.0*(x*x + y*y - 1.0)
    with tf.GradientTape() as outer_x, tf.GradientTape() as outer_y:
        outer_x.watch(x)
        outer_y.watch(y)
        with tf.GradientTape() as inner:
            inner.watch([t, x, y])
            p_ = p(t, x, y)
            #print(y)
            grad_p = inner.gradient(p_, [t, x, y])
        p_t = grad_p[0]
        p_x = grad_p[1]
        p_y = grad_p[2]
    p_xx = outer_x.gradient(p_x, x)
    p_yy = outer_y.gradient(p_y, y)
    a = (x*z + y) * p_x
    b = (y*z - x) * p_y
    c = (4.0*z - 8.0) * p_

    """
    v = tf.concat([t, x, y], axis = 1)
    for i, truth in enumerate(tf.math.is_inf(p_)):
        if truth:
            print(v[i], p(v[i][0]*one, v[i][1]*one, v[i][2]*one))
            #print('truncation_factor = ', p.truncation_factor)
    """
    #print(p_, p_t, p_xx)
    return  tf.reduce_mean((p_t + a + b + c - 0.5*(p_xx + p_yy))**2) #- 0.1 * tf.math.log(tf.reduce_sum(p_**2))
diff_op(*p.domain_sampler(0, 10))
#"""
# loss functions and domains
p.add_objective(diff_op, mean_square=False)
init_cond = lambda x, y: p(tf.zeros_like(x), x, y) - sp.standard_normal_2(x, y)
#print('init_cond ', init_cond(*p.domain_sampler(0, 100)))

p.add_objective(init_cond)
p.add_domain([space_x, space_y])
#p.add_objective(lambda t, theta: 1.0/p(t, tf.cos(theta), tf.sin(theta)))
#p.add_domain([time_, [-np.pi, np.pi]])
#p.add_domain([time_, space_x, space_y])
# learn the solution
p.solve_static(num_steps = 500, num_samples = [1000, 1000, 1], initial_rate = 0.001)#, decay_steps=500, decay_rate=0.1)
plotter = nnp.NNPlotter(funcs=[p], space=[space_x, space_y], num_pts_per_dim=30)
plotter.animate(file_path='../../images/{}.mp4'.format(model_name), t=time_, x_lim=space_x, y_lim=space_y, z_lim=None, num_frames=120)
p.save_weights('saved models/' + model_name)
#"""
print(quad.monte_carlo(p, p.principal_domain, time=2.7))
