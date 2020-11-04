import sys
sys.path.append("../")
from config import Config
import cupy as cp

def mean_square_error(y_pred, y_ture):
    return cp.asnumpy(cp.multiply(cp.mean(cp.sum(cp.square(cp.subtract(y_pred, y_ture)), 1)), 0.5))

def accuracy(y_pred, y_true):
    if len(y_true[0]) == 1:
        return cp.asnumpy(cp.mean(cp.equal(cp.where(y_pred<0.5, 0, 1), y_true).astype(cp.float32)))
    else:
        return cp.asnumpy(cp.mean(cp.equal(cp.argmax(y_pred, 1), cp.argmax(y_true, 1)).astype(cp.float32)))