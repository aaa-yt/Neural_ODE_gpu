import os
import json
from logging import getLogger
import numpy as np
import cupy as cp

from config import Config
from lib.helper import string_to_function, euler

logger = getLogger(__name__)

class NeuralODEModel:
    def __init__(self, config: Config):
        self.config = config
        self.dim_in = config.model.dim_in
        self.dim_out = config.model.dim_out
        self.max_time = config.model.max_time
        self.division = config.model.division
        # パラメータの初期値 = 0
        #alpha = np.zeros(shape=(self.division, self.dim_out), dtype=np.float32)
        #beta = np.zeros(shape=(self.division, self.dim_in, self.dim_in), dtype=np.float32)
        #gamma = np.zeros(shape=(self.division, self.dim_in), dtype=np.float32)

        # パラメータの初期値 = eps = 1e-8
        #eps = 1e-8
        #alpha = eps * np.ones(shape=(self.division, self.dim_out), dtype=np.float32)
        #beta = eps * np.ones(shape=(self.division, self.dim_in, self.dim_in), dtype=np.float32)
        #gamma = eps * np.ones(shape=(self.division, self.dim_in), dtype=np.float32)

        # パラメータの初期値 = 一様分布
        alpha = np.random.uniform(low=np.sqrt(3. / self.dim_out), high=np.sqrt(3. / self.dim_out), size=(self.division, self.dim_out)).astype(np.float32)
        beta = np.random.uniform(low=np.sqrt(3. / self.dim_in), high=np.sqrt(3. / self.dim_in), size=(self.division, self.dim_in, self.dim_in)).astype(np.float32)
        gamma = np.random.uniform(low=np.sqrt(3. / self.dim_in), high=np.sqrt(3. / self.dim_in), size=(self.division, self.dim_in)).astype(np.float32)

        self.params = (alpha, beta, gamma)
        self.A = np.eye(self.dim_out, self.dim_in, dtype=np.float32)
        self.function, self.d_function = string_to_function(config.model.function_type)
        self.t = np.linspace(0., self.max_time, self.division)
    
    def __call__(self, x0):
        def func(x, t, params, A, function, dim_in, division):
            index = int(t * (division - 1))
            x1 = x[:, :dim_in]
            y1 = cp.add(cp.dot(x1, params[1][index].T), params[2][index])
            y2 = cp.multiply(params[0][index], function(cp.dot(x1, A.T)))
            return cp.hstack([y1, y2])

        y0 = cp.zeros(shape=(len(x0), self.dim_out), dtype=cp.float32)
        x = euler(func, cp.hstack([x0, y0]), self.t, args=(self.params, self.A, self.function, self.dim_in, self.division))
        self.x = x[:, :, :self.dim_in]
        return x[-1, :, self.dim_in:]
    
    def gradient(self, x0, y_true):
        def func(a, t, params, A, function, bT, x, division):
            index = int(t * (division - 1))
            return cp.multiply(-1., cp.add(cp.dot(a, params[1][index]), cp.dot(cp.multiply(bT, cp.multiply(params[0][index], function(cp.dot(x[index], A.T)))), A)))

        n_data = len(x0)
        y_pred = self(x0)
        aT = cp.zeros_like(x0, dtype=cp.float32)
        bT = cp.divide(cp.subtract(y_pred, y_true), n_data)
        a = euler(func, aT, self.t[::-1], args=(self.params, self.A, self.d_function, bT, self.x, self.division))
        g_alpha = cp.sum(cp.multiply(bT, self.function(cp.dot(self.x, self.A.T))), 1)
        g_beta = cp.einsum("ilj,ilk->ijk", a[::-1], self.x)
        g_gamma = cp.sum(a[::-1], 1)
        return (g_alpha, g_beta, g_gamma)
    
    def load(self, model_path):
        if os.path.exists(model_path):
            logger.debug("loding model from {}".format(model_path))
            with open(model_path, "rt") as f:
                model_weights = json.load(f)
            alpha = np.array(model_weights.get("Alpha"))
            beta = np.array(model_weights.get("Beta"))
            gamma = np.array(model_weights.get("Gamma"))
            A = np.array(model_weights.get("A"))
            if self.params[0].shape != alpha.shape: alpha = self.params[0]
            if self.params[1].shape != beta.shape: beta = self.params[1]
            if self.params[2].shape != gamma.shape: gamma = self.params[2]
            self.params = (alpha, beta, gamma)
            if self.A.shape == A.shape: self.A = A
        self.params = tuple(cp.array(param) for param in self.params)
        self.A = cp.array(self.A)
        self.t = cp.array(self.t)
    
    def save(self, config_path, model_path):
        logger.debug("save model config to {}".format(config_path))
        self.config.save_parameter(config_path)
        logger.debug("save model to {}".format(model_path))
        model_data = {
            "Alpha": self.params[0].tolist(),
            "Beta": self.params[1].tolist(),
            "Gamma": self.params[2].tolist(),
            "A": self.A.tolist()
        }
        with open(model_path, "wt") as f:
            json.dump(model_data, f, indent=4)