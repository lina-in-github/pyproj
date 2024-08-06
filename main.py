import numpy as np

class SVM:
    def __init__(self, weights):
        self.weights = np.array(weights)

    def test(self, input_values):
        input_arr = np.array(input_values)
        return np.dot(self.weights, input_arr)

    def learn(self, err, rate):
        self.weights = [i+rate * err for i in self.weights]

class NetworkNode(SVM):
    def __init__(self, weights, func, deltafunc):
        super().__init__(weights)
        self.func = func
        self.deltafunc = deltafunc

    def test(self, input_values):
        result = super().test(input_values)
        return self.func(result)

    def learn(self, err, rate):
        # 计算反向传播的错误
        backprop_errors = []
        for weight in self.weights:
            backprop_error = weight * err * self.deltafunc(self.test(input_values))
            backprop_errors.append(backprop_error)
        # 更新权重
        super().learn(err, rate)
        return backprop_errors

# 示例激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 调用示例
network_node = NetworkNode([1, 2, 3], sigmoid, sigmoid_derivative)
input_values = [4, 5, 6]
error_values = [0.1, 0.2, 0.3]
rate = 0.5
for i in error_values:
    backprop_error_values = network_node.learn(i, rate)
print("反向传播的错误值:", backprop_error_values)