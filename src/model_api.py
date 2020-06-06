import os
import json
from datetime import datetime
from logging import getLogger
import numpy as np
import cupy as cp

from config import Config

logger = getLogger(__name__)

def start(config: Config):
    return ModelAPI(config).start()

class ModelAPI:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = None
    
    def start(self):
        tc = self.config.trainer
        batch_size = tc.batch_size
        self.model = self.load_model()
        self.dataset = self.load_dataset()
        y_pred = np.empty_like(self.dataset[1])
        for i in range(0, len(self.dataset[0]), batch_size):
            x_bs = cp.array(self.dataset[0][i:i+batch_size])
            y_bs = self.model(x_bs)
            y_pred[i:i+batch_size] = cp.asnumpy(y_bs)
        self.save_data_predict(cp.asnumpy(y_pred))
    
    def load_model(self):
        from model import NeuralODEModel
        model = NeuralODEModel(self.config)
        model.load(self.config.resource.model_path)
        return model
    
    def load_dataset(self):
        data_path = self.config.resource.data_path
        if os.path.exists(data_path):
            logger.debug("loading data from {}".format(data_path))
            with open(data_path, "rt") as f:
                datasets = json.load(f)
            x = datasets.get("Input")
            y = datasets.get("Output")
            if x is None or y is None:
                raise TypeError("Dataset does not exists in {}".format(data_path))
            if len(x[0]) != self.config.model.dim_in:
                raise ValueError("Input dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_in, len(x[0])))
            if len(y[0]) != self.config.model.dim_out:
                raise ValueError("Output dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_out, len(y[0])))
            return (np.array(x, dtype=np.float32), np.array(y,dtype=np.float32))
        else:
            raise FileNotFoundError("Dataset file can not loaded!")
    
    def save_data_predict(self, y_pred):
        rc = self.config.resource
        result_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(rc.result_dir, "result_predict_{}".format(result_id))
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "data_predict.json")
        data_predict = {
            "Input": self.dataset[0].tolist(),
            "Output": y_pred.tolist()
        }
        logger.debug("save prediction data to {}".format(result_path))
        with open(result_path, "wt") as f:
            json.dump(data_predict, f, indent=4)