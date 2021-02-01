# add required folders to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent.parent.parent.parent)
sys.path.insert(0, module_dir)

import tensorflow as tf
import numpy as np
from modules import nnplot as nnp
from modules import integrate as quad
from modules import utility as ut
from architecture import DGM
import json
import equation as eqn

# exponential ansatz
with open('../data/config.json') as config_file:
    config = json.load(config_file)
space_interval = np.array([0.0, config['radius']])

# define probability density
p = DGM(dim=1, num_nodes=config['num_nodes'], num_layers=config['num_layers'], dtype=getattr(tf, config['dtype']),\
        activation=getattr(tf.keras.activations, config['activation']), final_activation=getattr(tf.keras.activations, config['final_activation']),\
        data_path=config['data_path'], dpl_type=config['data_pipeline_type'])

#"""
try:
   p.load_weights()
except:
   pass
#"""

# define differential operator
@ut.timer
#@tf.function
def diff_op(r):
    return eqn.diff_op(p, r)

# add as an objective
p.add_objective(diff_op)
p.add_domain(space_interval, 'uniform')

# define initial condition
def bdry_cond(r):
    return eqn.bdry_cond(p)
p.add_objective(bdry_cond)
p.add_domain(space_interval, 'uniform')

p(*p.domains[0].sample(5))
p.summary()
#exit()

# learn the solution
"""
p.build_db(config['num_samples'], normalize=False)
final_loss = p.solve(epochs=config['num_epochs'], batch_size=config['batch_size'], initial_rate=config['initial_learning_rate'])
p.save_weights()
#"""
# define the Cartesian version of p
class Prob(tf.keras.models.Model):
    def __init__(self):
        super().__init__(dtype=p.dtype)
    def call(self, x, y):
        r = tf.sqrt(x*x + y*y)
        return tf.exp(-p(r))

prob = Prob()
normalizer = 2.0 * np.pi * quad.monte_carlo(prob, [config['radius'] * np.array([-1.0, 1.0])] * 2 , num_pts=10000)

class NormalizedProb(tf.keras.models.Model):
   def __init__(self):
       super().__init__(dtype=p.dtype)
   def call(self, x, y):
       r = tf.sqrt(x*x + y*y)
       return tf.exp(-p(r)) / normalizer

prob = NormalizedProb()
# plot and save the results
space_x = config['fraction_of_space_to_be_plotted'] * config['radius'] * np.array([-1.0, 1.0])
space_y = config['fraction_of_space_to_be_plotted'] * config['radius'] * np.array([-1.0, 1.0])
#"""
plotter = nnp.NNPlotter(funcs=[prob], space=[space_x, space_y], num_pts_per_dim=30)
plotter.plot(file_path='../data/solution.png', x_lim=space_x, y_lim=space_y, z_lim=None)
#"""
