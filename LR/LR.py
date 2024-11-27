
import random
import torch



def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


"""生成小批量样本"""
def data_Iter(batch_size,feature,lable):
    num_example=feature.shape[0]
    indices=list(range(num_example))
    random.shuffle(indices)

    for i in range(0,num_example,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_example)])
        yield feature[batch_indices],lable[batch_indices]

w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

def Liner(X,w,b):
    return torch.matmul(X,w)+b

def loss(y,y_pred):
    return (y_pred - y.reshape(y_pred.shape)) ** 2 / 2

def sgd(params,lr,batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():#不计算梯度
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()#将梯度清零

lr = 0.03
num_epochs = 3
batch_size = 10

for epoch in range(num_epochs):
    for X,y in data_Iter(10,features,labels):
        l=loss(Liner(X,w,b),y)

        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(Liner(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
print(w,b)