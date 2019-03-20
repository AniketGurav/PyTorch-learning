"""
Title: PyTorch/ Get Started/ a 60-min Blitz/ What is PyTorch
Main Author: PyTorch
Editor: Shengjie Xiu
Time: 2019/3/19
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

from __future__ import print_function
import torch,numpy as np

# 1. Tensors
# region Description

print('\nTensors\n')

#Construct a 5x3 matrix
x = torch.empty(5, 3)
print(x)

#Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)

#Construct a matrix filled zeros and of dtype long
x = torch.zeros(5, 3, dtype = torch.long)
print(x)

#Construct a tensor directly from data
x = torch.tensor([5.5, 3]) #前三种自动生成的也是tensor类型
print(x)

#or create a tensor based on an existing tensor. These methods
#will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
# endregion


# 2. Operations
# region Description
print('\nOperations\n')

#Addition: syntax 1
y = torch.rand(5, 3)
print(x + y)

#Addition: syntax 2
print(torch.add(x, y))

#Addition: providing an output tensor as argument
result = torch.empty([5,3])
torch.add(x, y, out=result)
print(result)

#Addition: in-place
y.add_(x)  # different to y.add()
print(y)

'''
Any operation that mutates a tensor in-place is post-fixed
with an _. For example: x.copy_(y), x.t_(), will change x.
'''

#You can use standard NumPy-like indexing with all bells and whistles!
print(x[:, 1])

#Resizing: If you want to resize/reshape tensor, you can use torch.view:
x = torch.rand([4,4])
y1 = x.view(16)
y2 = x.view(-1, 8) # the size -1 is inferred from other dimensions
print(x.size(), y1.size(), y2.size())
# endregion


# 3. NumPy Bridge
# region Description
print('\nNumpy Bridge\n')

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)

b = a.numpy()   # torch->numpy (here .numpy() is a function of torch)
print(b)

a.add_(1)
print(a)
print(b)

#Converting NumPy Array to Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a) # numpy->torch
np.add(a, 1, out=a)
print(a)
print(b)

'''
The Torch Tensor and NumPy array will share their 
underlying memory locations, and changing one will change the other.
'''
# endregion


# 4. CUDA Tensors
# region Description

print('\nCUDA Tensors\n')

# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda") # a CUDA device object 初始化所选设备
    y = torch.ones_like(x, device=device) # directly create a tensor on GPU
    x = x.to(device) # or just use strings ``.to("cuda")`` 将数据发到GPU
    z_GPU = x + y
    print(z_GPU)
    z_CPU = z_GPU.to("cpu", torch.double) # 将数据发到CPU ``.to`` can also change dtype together!
    print(z_CPU)

'''
Tensors can be moved onto any device using the .to method.
'''
# endregion



