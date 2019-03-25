"""
Title: 莫烦/ 建造第一个神经网络/ Lesson5-批训练
Main Author: Morvan Zhou
Editor: Shengjie Xiu
Time: 2019/3/22
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)


# 定义dataset：将上面的数据打包进dataset中
# Dataset wrapping tensors
torch_dataset = Data.TensorDataset(x, y)


# 定义loader=批
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(
                loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())

if __name__ == '__main__':   # mark as the main function that when this .py is excuted, this section will function; but when this .py is inherited, this section will not function
    show_batch()

'''
>>>seq = ['one', 'two', 'three']
>>> for i, element in enumerate(seq):
...     print i, element
...
0 one
1 two
2 three
'''
