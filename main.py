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
        return sum([i*j for i,j in zip(self.weights, input_arr)])

    def learn(self, err, rate):
        self.weights = [i+rate * err for i in self.weights]

class NetworkNode(SVM):
    def __init__(self, weights,subNetworks=[],rootNetworks=[],brotherNetworks=[],func=relu, deltafunc=delta_relu):
        super().__init__(weights)
        self.func = func
        self.deltafunc = deltafunc
        self.SN=subNetworks
        self.RN=rootNetworks
        self.BN=brotherNetworks
    @classmethod
    def network_init(cls,inputnum,*layernum):
        weightn=inputnum
        network=[]
        for i in layernum:
            k:list[NetworkNode]=[]
            q=[]
            for j in range(i):
                k.append(NetworkNode([random.uniform(-1,1) for _ in range(weightn)][:],subNetworks=q))
            for j in q:
                j.RN=k
                j.BN=q
            network.append(k)
            weightn=i
            k=q
        return(network[0],network[-1])
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
        t=[0 for i in self.BN]
        for i,j in zip(self.BN,err):
            p=i.learn(j,rate)
            for k,j in enumerate(p):
                t[k]+=j
        for i,j,k in zip(self.SN,t,range(len(t))):
            i.backward(j,rate)
# 测试神经网络
if __name__ == "__main__":
    # 初始化网络，假设我们有一个输入层，两个隐藏层和输出层
    # 输入层有3个神经元，第一个隐藏层有4个神经元，第二个隐藏层有3个神经元，输出层有2个神经元
    layers = [3, 4, 3, 2]
    inputnum = 3  # 输入特征数量
    network, output_layer = NetworkNode.network_init(inputnum, *layers)
    
    # 随机生成一些输入数据进行测试
    test_inputs = np.array([random.uniform(-1, 1) for _ in range(inputnum)])
    
    # 测试网络的前向传播
    network[0].forward(test_inputs)
    
    # 测试输出层的输出
    for node in output_layer:
        print(f"Output of node {node}: {node.test(test_inputs)}")
        print(node.weights)

    # 假设我们有一个目标输出，这里我们使用随机数作为示例
    target_output = np.array([random.uniform(-1, 1) for _ in range(len(output_layer))])
    
    # 计算输出层的误差
    errors = [node.test(test_inputs) - target for node, target in zip(output_layer, target_output)]
    
    # 学习率
    learning_rate = 0.01
    
    # 反向传播并更新权重
    output_layer[0].backward(errors, learning_rate)
    
    # 再次测试网络的输出，以查看权重更新后的效果
    print("\nAfter learning:")
    for node in output_layer:
        print(f"Output of node {node}: {node.test(test_inputs)}")