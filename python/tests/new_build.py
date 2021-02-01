# add required folders to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
image_dir = str(script_dir.parent.parent)
print(image_dir)
sys.path.insert(0, module_dir)
from modules import nnsolver2 as nns
from modules import lstm
import numpy as np
import tensorflow as tf
"""
dp = nns.DataPipelineCSV('../../data')
dp.add_domain([-6., 6.], 'uniform', [-6., 6.], 'uniform')
dp.add_domain([-6., 6.], 'uniform', [-6., 6.], 'uniform')
dp.build_db(num_pts=300, normalize=True)
dp.open_db()
print(dp.read_db(num_pts=5, start=10))
print(dp.read_db(num_pts=5, start=100))
dp.close_db()
#"""
#"""
# exponential ansatz
D = 0.51
R = 6.0
T = 10.0
alpha = 0.3
beta = 3.0
gamma = 1.0
space_r = np.array([0.0, R])
space_theta = np.array([-np.pi, np.pi])
space_x = gamma * R * np.array([-1., 1.])
space_y = gamma * R * np.array([-1., 1.])
time_ = np.array([0.0, T])
num_nodes = 50
num_layers = 3
nn_type = 'LSTMForget'
model_name = 'cricle13_{}_{}_{}_{}'.format(num_nodes, num_layers, nn_type, str(D).replace('.', '_'))
f = lstm.LSTMForget(10, 3, dpl_type='csv', name='test_nn')
# define differential operator
#@ut.timer
def diff_op(t, r):
    r2 = r*r
    z = 4.0*(r2 - 1.0)
    #t = t[0]* tf.ones_like(r)
    with tf.GradientTape() as outer_r:
        outer_r.watch(r)
        with tf.GradientTape() as inner:
            inner.watch([t, r])
            f_ = f.eval(t, r)
        grad_f = inner.gradient(f_, [t, r])
        f_t = grad_f[0]
        f_r = grad_f[1]
    f_rr = outer_r.gradient(f_r, r)
    b = f_r
    a = (D + z*r2) * b
    c = 4.0*r*(z + 2.0)
    eqn = - r*f_t + a - c + D * r * (f_rr - b**2)
    return eqn

# add as an objective
f.add_objective(diff_op)
f.add_domain(time_, 'uniform', space_r, 'uniform')

# define initial condition
log_4_R2 = tf.cast(2.0 * tf.math.log(2.0 * R), f.dtype)
init_cond = lambda r: f.eval(tf.zeros_like(r), r) - 0.5*r*r
#exit()

# add as an objective
f.add_objective(init_cond)
f.add_domain(space_r, 'uniform')

# define boundary condition
def bdry_cond(t):
    r = R*tf.ones_like(t)
    with tf.GradientTape() as tape:
        tape.watch(r)
        p = tf.exp(-f.eval(t, r))
    dp_dr = tape.gradient(p, r)
    expr = 4.0*R*(R**2-1.0)*p + D*R*dp_dr
    return expr

# add as an objective
f.add_objective(bdry_cond)
f.add_domain(time_, 'uniform')
#"""
# learn the solution
f.build_db(10000, normalize=False)
#"""

f.dpl.open_db()
#print(lf.dpl.read_db(num_pts=5, start=10))
#print(lf.dpl.read_db(num_pts=5, start=100))
print('fuyioh', f(f.dpl.read_db(num_pts=5, start=100)))
f.compile(optimizer="Adam", loss="mse", metrics=["mae"])
x_train = f.dpl.read_db(num_pts=10)
print(x_train)
print(f(x_train))
y_train = tf.zeros_like(x_train)
f.fit(x_train, y_train, epochs=10)
#"""
