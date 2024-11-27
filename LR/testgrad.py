import torch

x=torch.tensor([1.0,3.0,5.0],requires_grad=True)
y=torch.matmul(x.t(),x)
z=2*x
y.backward()
x.grad.zero_()
z.sum().backward()
print(x.grad)