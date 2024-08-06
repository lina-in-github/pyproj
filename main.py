import random
class SVM:
    def __init__(self,bias,*weights,function=lambda x:x if x>0 else 0,deltafunc=lambda x:0 if x<0 else 1) -> None:
        self.bias=bias
        self.weights=weights
        self.input_cache=[0 for i in weights]
        self.output_cache=0
        self.func=function
        self.dfunc=deltafunc
        self.deltaweights=[0 for i in weights]
    def output(self,*input):
        assert len(input)==len(self.weights)
        self.input_cache=input
        self.output_cache=sum([k*x for k,x in zip(input,self.weights)])+self.bias
        return self.func(self.output_cache)
    def learn(self,derr:float,rate=0.005):
        target=derr
        self.deltaweights=[-i-j*target*rate*self.dfunc(self.output_cache) for i,j in zip(self.deltaweights,self.input_cache)]
        return [i*target*rate*self.dfunc(self.output_cache) for i,j in zip(self.weights,self.input_cache)]
    def update(self):
        self.weights=[i+j for i,j in zip(self.weights,self.deltaweights)]
t=SVM(*[random.random() for i in range(3)])
for _ in range(10):
    for i in range(0,100):
        for j in range(0,100):
            v=t.learn(t.output(0.1*i,0.1*j)-0.2*i+0.3*j)
            print(v) 
    t.update()           
print("w"+str(t.weights))