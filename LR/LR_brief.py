from random import shuffle
import torch.nn as nn
import torch
import random

from torch.nn import MSELoss
from torch.utils import data
from LR import synthetic_data

def load_array(data_array,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_array)#将数据变为tensordataset
    return data.DataLoader(dataset,batch_size,shuffle=is_train)#是可迭代对象

if __name__=='__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)


    batch_size=10
    data_iter=load_array((features,labels),batch_size)
    model=nn.Sequential(nn.Linear(2,1))
    model[0].weight.data.normal_(0,0.01)
    model[0].bias.data.fill_(0)

    loss=nn.MSELoss()
    trainer=torch.optim.SGD(model.parameters(),lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(model(X) ,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(model(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

