"""
Title: PyTorch/ Get Started/ a 60-min Blitz/ Neural Network
Main Author: PyTorch
Editor: Shengjie Xiu
Time: 2019/3/20
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1.Define the network
# region Description

print('\nDefine the network\n')


class Net(nn.Module):   # Net继承了nn.Module类

    def __init__(self):  # 构造函数，在类的一个对象被建立时，马上运行
        super(Net, self).__init__()  # super也是一个定义好的类
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)   # con1/conv2...都是类的attribute
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 成员函数与一般函数区别是多了self
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()   # 创建Net类对象
print(net)

# 有10个params：5个权重＋5个输出特征数（conv2d的filters数/fc的输出神经元个数）
params = list(net.parameters())  # parameters是Module类的关键attribute
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Zero the gradient buffers of all parameters
net.zero_grad()
# backprops with random gradients
out.backward(torch.randn(1, 10))

'''
You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
'''

'''
# Recap:
# torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
# nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
# nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
# autograd.Function - Implements forward and backward definitions of an autograd operation. Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.
'''
# endregion


# 2. Loss Function
# region Description

print('\nLoss Function\n')

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# 此时我们的BP可以从loss的值开始，而不是任意定义的了
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
# endregion


# 3. Backprop
# region Description

print('\nBackprop\n')

net.zero_grad()  # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()   # Backprop

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# endregion


# 4. update the weights
# region Description

print('\nupdate the weights\n')

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # Does the update
# endregion
