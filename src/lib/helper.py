import cupy as cp

from .functions import Sigmoid, Relu

def string_to_function(function_type):
    all_functions = {
        "sigmoid": Sigmoid(),
        "relu": Relu()
    }
    if function_type.lower() in all_functions:
        function_type = function_type.lower()
        return all_functions[function_type], all_functions[function_type].derivative
    else:
        return all_functions["sigmoid"], all_functions["sigmoid"].derivative

def euler(func, x0, t, args=None):
    solution = cp.empty(shape=(len(t), len(x0), len(x0[0])))
    solution[0] = x0
    x = x0
    for i, dt in enumerate(cp.diff(t)):
        x = cp.add(x, cp.multiply(dt, func(x, t[i], *args)))
        solution[i+1] = x
    return solution