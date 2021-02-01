# defines various pdfs for nnsolvers

import tensorflow as tf
import numpy as np

def normal(x, *params):
    mu, sigma = params
    return tf.exp(-0.5 * ((x - mu)/sigma)**2)/(tf.sqrt(2.0*np.pi) * sigma)


class Density:
    def __init__(self, pdf, param_details):
        self.pdf = pdf
        self.param_details = param_details
        self.num_params = len(param_details)

class Normal(Density):
    def __init__(self):
        super(Normal, self).__init__(pdf = normal, param_details = [{}, {'positive': True}])
