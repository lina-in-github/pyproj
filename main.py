import numpy as np
import random
def relu(x):
    return x if x>0 else 0

def delta_relu(x):
    return 1 if x>0 else 0

class SVM:
    def __init__(self, weights):
        self.weights = np.array(weights)

    def test(self, input_values):
        input_arr = np.array(input_values)
        return np.dot(self.weights, input_arr)

    def learn(self, err, rate):
        self.weights = [i+rate * err for i in self.weights]

class NetworkNode(SVM):
    def __init__(self, weights,subNetworks:list["NetworkNode"]=[],rootNetworks:list["NetworkNode"]=[],brotherNetworks:list["NetworkNode"]=[],func=relu, deltafunc=delta_relu):
        super().__init__(weights)
        self.func = func
        self.deltafunc = deltafunc
        self.SN=subNetworks
        self.RN=rootNetworks
        self.BN=brotherNetworks
    @classmethod
    def network_init(cls,inputnum,*layernum)->tuple[list["NetworkNode"],list["NetworkNode"]]:
        weightn=inputnum
        network=[]
        for i in layernum:
            k:list[NetworkNode]=[]
            q=[]
            for j in range(i):
                k.append(NetworkNode([random.uniform(-1,1) for _ in weightn],subNetworks=q))
            for j in q:
                j.RN=k
                j.BN=q
            network.append(k)
            weightn=i
            k=q
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
    def forward(self,data):
        p=[]
        for i in self.BN:
            p.append(i.test(data))
        for i in self.RN:
            i.forward(p)
    def backward(self,err,rate):
        t=[0 for i in len(self.BN)]
        for i in self.BN:
            p=i.learn(err,rate)
            for k,j in enumerate(p):
                t[k]+=j
        for i,j,k in zip(self.SN,t,range(len(t))):
            i.backward(j,rate)

# 调用示例
network_node = NetworkNode([1, 2, 3])
input_values = [4, 5, 6]
error_values = [0.1, 0.2, 0.3]
rate = 0.5
for i in error_values:
    backprop_error_values = network_node.learn(i, rate)
print("反向传播的错误值:", backprop_error_values)