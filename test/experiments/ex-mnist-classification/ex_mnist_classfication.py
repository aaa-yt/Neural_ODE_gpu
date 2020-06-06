import os
import shutil
import json
import configparser
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def create_config_file(config, config_path):
    if os.path.exists(config_path): return
    config_parser = configparser.ConfigParser()
    config_parser["MODEL"] = {
        "Input_dimension": config["Input_dimension"],
        "Output_dimension": config["Output_dimension"],
        "Maximum_time": config["Maximum_time"],
        "Weights_division": config["Weights_division"],
        "Function_type": config["Function_type"]
    }
    config_parser["TRAINER"] = {
        "Optimizer_type": config["Optimizer_type"],
        "Learning_rate": config["Learning_rate"],
        "Momentum": config["Momentum"],
        "Decay": config["Decay"],
        "Decay2": config["Decay2"],
        "Epoch": config["Epoch"],
        "Batch_size": config["Batch_size"],
        "Test_size": config["Test_size"],
        "Validation_size": config["Validation_size"],
        "Is_visualize": config["Is_visualize"],
        "Is_accuracy": config["Is_accuracy"]
    }
    with open(config_path, "wt") as f:
        config_parser.write(f)

def create_data_file(config, data_path):
    if os.path.exists(data_path): return
    import chainer
    train, test = chainer.datasets.get_mnist()
    x_train, y_train = train._datasets
    x_test, y_test = test._datasets
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    n_labels = len(np.unique(y))
    y = np.eye(n_labels)[y]
    dataset = {
        "Input": x.tolist(),
        "Output": y.tolist()
    }
    with open(data_path, "wt") as f:
        json.dump(dataset, f, indent=4)

def setting_file(path):
    if not os.path.exists(path["Config_dir"]):
        os.makedirs(path["Config_dir"])
    if not os.path.exists(path["Data_dir"]):
        os.makedirs(path["Data_dir"])
    if not os.path.exists(path["Data_processed_dir"]):
        os.makedirs(path["Data_processed_dir"])
    shutil.copy(path["Ex_config_path"], path["Config_path"])
    shutil.copy(path["Ex_data_path"], path["Data_path"])

def copy_result(path):
    if not os.path.exists(path["Ex_result_dir"]):
        os.makedirs(path["Ex_result_dir"])
    for p in os.listdir(path["Result_dir"]):
        result_dir = os.path.join(path["Result_dir"], p)
        for pp in os.listdir(result_dir):
            result_path = os.path.join(result_dir, pp)
            shutil.move(result_path, path["Ex_result_dir"])
    shutil.move(path["Model_path"], path["Ex_dir"])

def clear(path):
    shutil.rmtree(path["Data_dir"])
    shutil.rmtree(path["Config_dir"])
    shutil.rmtree(path["Model_dir"])

if __name__ == "__main__":
    config = {
        "Input_dimension": 784,
        "Output_dimension": 10,
        "Maximum_time": 1.0,
        "Weights_division": 50,
        "Function_type": "relu",
        "Optimizer_type": "rmsprop",
        "Learning_rate": 0.01,
        "Momentum": 0.9,
        "Decay": 0.99,
        "Decay2": 0.999,
        "Epoch": 5,
        "Batch_size": 128,
        "Test_size": 0.1,
        "Validation_size": 0.1,
        "Is_visualize": 0,
        "Is_accuracy": 1
    }

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_dir, "data")
    data_processed_dir = os.path.join(data_dir, "processed")
    data_path = os.path.join(data_processed_dir, "data.json")
    config_dir = os.path.join(project_dir, "config")
    config_path = os.path.join(config_dir, "parameter.conf")
    model_dir = os.path.join(project_dir, "model")
    model_path = os.path.join(model_dir, "model.json")
    result_dir = os.path.join(data_dir, "result")
    program_path = os.path.join(os.path.join(project_dir, "src"), "run.py")
    test_dir = os.path.join(project_dir, "test")
    ex_dir = os.path.join(os.path.join(test_dir, "experiments"), "ex-mnist-classification")
    ex_data_path = os.path.join(ex_dir, "data.json")
    ex_config_path = os.path.join(ex_dir, "parameter.conf")
    ex_result_dir = os.path.join(ex_dir, "result")
    data_predict_path = os.path.join(ex_result_dir, "data_predict.json")

    path = {
        "Project_dir": project_dir,
        "Data_dir": data_dir,
        "Data_processed_dir": data_processed_dir,
        "Data_path": data_path,
        "Config_dir": config_dir,
        "Config_path": config_path,
        "Model_dir": model_dir,
        "Model_path": model_path,
        "Result_dir": result_dir,
        "Program_path": program_path,
        "Test_dir": test_dir,
        "Ex_dir": ex_dir,
        "Ex_data_path": ex_data_path,
        "Ex_config_path": ex_config_path,
        "Ex_result_dir": ex_result_dir,
        "Data_predict_path": data_predict_path
    }

    #subprocess.call(["python", "-m", "pip", "install", "chainer"])
    create_config_file(config, ex_config_path)
    print("check1")
    create_data_file(config, ex_data_path)
    print("check2")
    setting_file(path)
    print("check3")
    subprocess.call(["python", program_path, "train"])
    shutil.copy(os.path.join(os.path.join(model_dir, os.listdir(model_dir)[0]), "model.json"), model_path)
    subprocess.call(["python", program_path, "predict"])
    copy_result(path)
    clear(path)