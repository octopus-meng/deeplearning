import torch
import torchvision
from sympy.core.random import shuffle
from torch.utils import data
from torchvision import transforms

epsilon=0.0001
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
batch_size=256
dataIter=data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
input_size=784
output_size=10
W=torch.normal(0,0.01,size=(input_size,output_size),requires_grad=True)
b=torch.zeros(output_size,requires_grad=True)

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def softmax(X):
    exp_x=torch.exp(X)
    exp_x_sum=exp_x.sum(1,keepdim=True)
    return exp_x/exp_x_sum

def model(X):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)

def cross_entropy(y,y_pred):
    return -torch.log(y_pred[[range(len(y))],y]+epsilon)

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(model, data_iter):
    metric=Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(model(X),y),len(y))
    return metric[0]/metric[1]


trainer=torch.optim.SGD([W,b],lr=0.003)
epoch=5

for i in range(epoch):
    for X,y in dataIter:
        l=cross_entropy(y,model(X))
        trainer.zero_grad()
        l.sum().backward()
        trainer.step()



test_Iter=data.DataLoader(mnist_test,batch_size=batch_size)
print(evaluate_accuracy(model,test_Iter))











