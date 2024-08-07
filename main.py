import numpy as np
import random
from abc import ABC,abstractmethod as abstract
def relu(x):
    return x if x>0 else 0

def delta_relu(x):
    return 1 if x>0 else 0
@ABC 
class Function:
    @abstract
    def test(self,i):...
    @abstract
    def learn(self,o,i):...
class ReLUFunc(Function):
    def test(self,x):return relu(x)
    def learn(self,y,i):return y*delta_relu(i)
class SVM(Function):
    def __init__(self, weights):
        self.weights = np.array(weights)

    def test(self, input_values):
        input_arr = np.array(input_values)
        return sum([i*j for i,j in zip(self.weights, input_arr)])

    def learn(self, err, input_values,rate=0.05):
        self.weights = [i+rate * err for i in self.weights]

class NetworkNode(SVM):
    def __init__(self, weights,func=ReLUFunc):
        super().__init__(weights)
        self.func = func
    def test(self, input_values):
        result = super().test(input_values)
        return self.func(result)
    def learn(self, err, input_values, rate=0.05):
        # 计算反向传播的错误
        backprop_errors = []
        for weight in self.weights:
            backprop_error = err * self.func.learn(weight,self.test(input_values))
            backprop_errors.append(backprop_error)
        # 更新权重
        super().learn(err, rate)
        return backprop_errors
class NetworkLayer():
    def __init__(self,inputnum,outputnum):...