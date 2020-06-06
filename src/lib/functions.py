import cupy as cp

class Sigmoid:
    def __init__(self, gain=1.):
        self.gain = gain
    
    def __call__(self, x):
        return self._sigmoid(x)
    
    def derivative(self, x):
        return self._derivative_sigmoid(x)
    
    def _sigmoid(self, x):
        return cp.divide(1., cp.add(1., cp.exp(cp.multiply(-self.gain, x))))
    
    def _derivative_sigmoid(self, x):
        return cp.multiply(self.gain, cp.multiply(self._sigmoid(x), cp.subtract(1., self._sigmoid(x))))

class Relu:
    def __init__(self, gain=1.):
        self.gain = gain
    
    def __call__(self, x):
        return self._relu(x)
    
    def derivative(self, x):
        return self._derivative_relu(x)
    
    def _relu(self, x):
        return cp.where(x < 0, 0, cp.multiply(self.gain, x))
    
    def _derivative_relu(self, x):
        return cp.where(x < 0, 0, self.gain)