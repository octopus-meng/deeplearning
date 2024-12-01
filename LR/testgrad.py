import torch


x=torch.tensor([[1.0,3.0,5.0],[2.0,4.0,6.0]],requires_grad=True)
print(x[[0,1],[2,1]])
'''
y=torch.matmul(x.t(),x)
z=2*x
y.backward()
x.grad.zero_()
z.sum().backward()
print(x.grad)
'''