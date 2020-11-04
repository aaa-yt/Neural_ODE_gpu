import sys
sys.path.append("../")
from config import Config
import numpy as np
import cupy as cp

class Optimizer(object):
    def __init__(self, config: Config):
        self.config = config
        self.mc = config.model
        self.tc = config.trainer

class SGD(Optimizer):
    def __init__(self, config: Config):
        super(SGD, self).__init__(config)
        self.rate = self.tc.rate
    
    def __call__(self, params, g_params):
        return tuple(cp.subtract(param, cp.multiply(self.rate, g_param)) for param, g_param in zip(params, g_params))
    

class Momentum(Optimizer):
    def __init__(self, config: Config):
        super(Momentum, self).__init__(config)
        self.rate = self.tc.rate
        self.momentum = self.tc.momentum
        alpha = cp.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=cp.float32)
        beta = cp.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=cp.float32)
        gamma = cp.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=cp.float32)
        self.v = (alpha, beta, gamma)
    
    def __call__(self, params, g_params):
        new_params = tuple(cp.add(param, cp.subtract(cp.multiply(self.momentum, cp.subtract(param, v)), cp.multiply(self.rate, g_param))) for param, g_param, v in zip(params, g_params, self.v))
        self.v = params
        return new_params

class AdaGrad(Optimizer):
    def __init__(self, config: Config):
        super(AdaGrad, self).__init__(config)
        self.rate = self.tc.rate
        alpha = cp.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=cp.float32)
        beta = cp.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=cp.float32)
        gamma = cp.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=cp.float32)
        self.v = (alpha, beta, gamma)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(cp.subtract(param, cp.multiply(cp.divide(self.rate, cp.sqrt(cp.add(cp.add(v, cp.square(g_param)), self.eps).astype(cp.float32))), g_param)), cp.add(v, cp.square(g_param))) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params

class RMSprop(Optimizer):
    def __init__(self, config: Config):
        super(RMSprop, self).__init__(config)
        self.rate = self.tc.rate
        self.decay = self.tc.decay
        alpha = cp.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=cp.float32)
        beta = cp.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=cp.float32)
        gamma = cp.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=cp.float32)
        self.v = (alpha, beta, gamma)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(cp.subtract(param, cp.multiply(cp.divide(self.rate, cp.sqrt(cp.add(cp.add(cp.multiply(self.decay, v), cp.multiply(cp.subtract(1., self.decay), cp.square(g_param))), self.eps).astype(cp.float32))), g_param)), cp.add(cp.multiply(self.decay, v), cp.multiply(cp.subtract(1., self.decay), cp.square(g_param)))) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


class AdaDelta(Optimizer):
    def __init__(self, config: Config):
        super(AdaDelta, self).__init__(config)
        self.decay = self.tc.decay
        alpha = cp.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=cp.float32)
        beta = cp.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=cp.float32)
        gamma = cp.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=cp.float32)
        self.v = (alpha, beta, gamma)
        self.s = (alpha, beta, gamma)
        self.params_prev = (alpha, beta, gamma)
        self.eps = 1e-8


def get(config: Config):
    optimizer_type = config.trainer.optimizer_type.lower()
    if optimizer_type == "momentum":
        return Momentum(config)
    elif optimizer_type == "adagrad":
        return AdaGrad(config)
    elif optimizer_type == "rmsprop":
        return RMSprop(config)
    else:
        return SGD(config)

    '''
    tc = config.trainer
    all_optimizer = {
        "sgd": SGD(config=config),
        "momentum": Momentum(config=config),
        "adagrad": AdaGrad(config=config),
        "rmsprop": RMSprop(config=config)
    }

    optimizer_type = tc.optimizer_type
    if optimizer_type.lower() in all_optimizer:
        optimizer_type = optimizer_type.lower()
        return all_optimizer[optimizer_type]
    else:
        return all_optimizer["sgd"]
    '''