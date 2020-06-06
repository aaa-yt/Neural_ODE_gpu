import os
import configparser
from logging import getLogger

logger = getLogger(__name__)

def _project_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _data_dir():
    return os.path.join(_project_dir(), "data")

def _model_dir():
    return os.path.join(_project_dir(), "model")


class Config:
    def __init__(self):
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.trainer = TrainerConfig()
    
    def load_parameter(self):
        if os.path.exists(self.resource.config_path):
            logger.debug("loading parameter from {}".format(self.resource.config_path))
            config_parser = configparser.ConfigParser()
            config_parser.read(self.resource.config_path, encoding='utf-8')
            read_model = config_parser['MODEL']
            if read_model.get("Input_dimension") is not None: self.model.dim_in = int(read_model.get("Input_dimension"))
            if read_model.get("Output_dimension") is not None: self.model.dim_out = int(read_model.get("Output_dimension"))
            if read_model.get("Maximum_time") is not None: self.model.max_time = float(read_model.get("Maximum_time"))
            if read_model.get("Weights_division") is not None: self.model.division = int(read_model.get("Weights_division"))
            if read_model.get("Function_type") is not None: self.model.function_type = read_model.get("Function_type")
            read_trainer = config_parser['TRAINER']
            if read_trainer.get("Optimizer_type") is not None: self.trainer.optimizer_type = read_trainer.get("Optimizer_type")
            if read_trainer.get("Learning_rate") is not None: self.trainer.rate = float(read_trainer.get("Learning_rate"))
            if read_trainer.get("Momentum") is not None: self.trainer.momentum = float(read_trainer.get("Momentum"))
            if read_trainer.get("Decay") is not None: self.trainer.decay = float(read_trainer.get("Decay"))
            if read_trainer.get("Decay2") is not None: self.trainer.decay2 = float(read_trainer.get("Decay2"))
            if read_trainer.get("Epoch") is not None: self.trainer.epoch = int(read_trainer.get("Epoch"))
            if read_trainer.get("Batch_size") is not None: self.trainer.batch_size = int(read_trainer.get("Batch_size"))
            if read_trainer.get("Test_size") is not None: self.trainer.test_size = float(read_trainer.get("Test_size"))
            if read_trainer.get("Validation_size") is not None: self.trainer.validation_size = float(read_trainer.get("Validation_size"))
            if read_trainer.get("Is_visualize") is not None: self.trainer.is_visualize = bool(int(read_trainer.get("Is_visualize")))
            if read_trainer.get("Is_accuracy") is not None: self.trainer.is_accuracy = bool(int(read_trainer.get("Is_accuracy")))
    
    def save_parameter(self, config_path):
        config_parser = configparser.ConfigParser()
        config_parser["MODEL"] = {
            "Input_dimension": self.model.dim_in,
            "Output_dimension": self.model.dim_out,
            "Maximum_time": self.model.max_time,
            "Weights_division": self.model.division,
            "Function_type": self.model.function_type
        }
        config_parser["TRAINER"] = {
            "Optimizer_type": self.trainer.optimizer_type,
            "Learning_rate": self.trainer.rate,
            "Momentum": self.trainer.momentum,
            "Decay": self.trainer.decay,
            "Decay2": self.trainer.decay2,
            "Epoch": self.trainer.epoch,
            "Batch_size": self.trainer.batch_size,
            "Test_size": self.trainer.test_size,
            "Validation_size": self.trainer.validation_size,
            "Is_visualize": int(self.trainer.is_visualize),
            "Is_accuracy": int(self.trainer.is_accuracy)
        }
        with open(config_path, "wt") as f:
            config_parser.write(f)


class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())
        self.data_processed_dir = os.path.join(self.data_dir, "processed")
        self.data_path = os.path.join(self.data_processed_dir, "data.json")
        self.model_dir = os.environ.get("MODEL_DIR", _model_dir())
        self.model_path = os.path.join(self.model_dir, "model.json")
        self.result_dir = os.path.join(self.data_dir, "result")
        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.config_dir = os.path.join(self.project_dir, "config")
        self.config_path = os.path.join(self.config_dir, "parameter.conf")
    
    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.data_processed_dir, self.model_dir, self.result_dir, self.log_dir, self.config_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
            

class ModelConfig:
    def __init__(self):
        self.dim_in = 1
        self.dim_out = 1
        self.max_time = 1.
        self.division = 100
        self.function_type = "sigmoid"


class TrainerConfig:
    def __init__(self):
        self.optimizer_type = "SGD"
        self.rate = 0.01
        self.momentum = 0.9
        self.decay = 0.99
        self.decay2 = 0.999
        self.epoch = 1
        self.batch_size = 10
        self.test_size = 0.2 #テストデータ = 全データ * test_size
        self.validation_size = 0.2 # バリデーションデータ = 全データ * (1 - test_size) * validation_size
        self.is_visualize = True
        self.is_accuracy = False