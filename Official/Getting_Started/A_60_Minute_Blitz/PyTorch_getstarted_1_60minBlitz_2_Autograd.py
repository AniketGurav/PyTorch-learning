"""
Title: PyTorch/ Get Started/ a 60-min Blitz/ Autograd: Automatic Differentiation
Main Author: PyTorch
Editor: Shengjie Xiu
Time: 2019/3/20
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

from __future__ import print_function
import torch
import numpy as np

# 1. Tensor
# region Description

print('\nTensor\n')

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z)
print(out)

# .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place.
#  The input flag defaults to False if not given.
# 如果没有在定义tensor时表明requires_grad，则需要grad时使用requires_grad_(True)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# endregion

# 2. Gradients
# region Description

print('\nGradients\n')

out.backward()
print('x grad:', x.grad)

# vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)   # backward里的值相当于上游的梯度矩阵，默认是size相同的ones矩阵
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

# You can also stop autograd from tracking history on Tensors
# with ``.requires_grad=True`` by wrapping the code block in
# ``with torch.no_grad():``
with torch.no_grad():
    print((x ** 2).requires_grad)
# endregion
