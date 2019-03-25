"""
Title: 莫烦/ 建造第一个神经网络/ Lesson4-保存提取
Main Author: Morvan Zhou
Editor: Shengjie Xiu
Time: 2019/3/22
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

import torch
import matplotlib.pyplot as plt


# fake data
# x data (tensor), shape=(100, 1)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# noisy y data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


def train():
    # 用快速搭建法 建神经网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    # 优化方法，损失函数
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    # 迭代训练
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot net1 result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    return net1


def save(net):
    # save net1
    # 2 ways to save the net
    torch.save(net, 'net.pkl')  # 1)save entire net
    # 2)save only the parameters
    torch.save(net.state_dict(), 'net_params.pkl')


def restore_net():
    # 1)restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    # plot net2 result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    # 2)restore only the parameters in net1 to net3, 需要先建立与参数原模型一样的模型，才能够导入
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # copy net1's parameters into net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    # plot net3 result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


net1 = train()

save(net1)

restore_net()

restore_params()
