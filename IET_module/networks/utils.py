#!/usr/bin/python3

import numpy as np

def null_activation(x):
    return x

def _fanin_init(tensor, alpha = 0):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    # bound = 1. / np.sqrt(fan_in)
    bound = np.sqrt( 1. / ( (1 + alpha * alpha ) * fan_in) )
    return tensor.data.uniform_(-bound, bound)

def _uniform_init(tensor, param=3e-3):
    return tensor.data.uniform_(-param, param)

def _constant_bias_init(tensor, constant = 0.1):
    tensor.data.fill_( constant )

def _normal_init(tensor, mean=0, std =1e-3):
    return tensor.data.normal_(mean,std)

def layer_init(layer, weight_init = _fanin_init, bias_init = _constant_bias_init ):
    weight_init(layer.weight)
    bias_init(layer.bias)

def basic_init(layer):
    layer_init(layer, weight_init = _fanin_init, bias_init = _constant_bias_init)

def uniform_init(layer):
    layer_init(layer, weight_init = _uniform_init, bias_init = _uniform_init )