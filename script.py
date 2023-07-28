import torch

x = torch.randn((3,4), requires_grad=True)
v = torch.ones_like(x, dtype=torch.float32)
e = torch.randn((3,4))
y = x * x[:, 0].reshape(-1,1)
print(x)
#v = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float) # stand-in for gradients
print((torch.autograd.grad(y, x, e)[0]*e).view(x.shape[0], -1).sum(dim=1))

#print(x.grad, x)