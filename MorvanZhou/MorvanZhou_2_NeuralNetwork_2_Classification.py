"""
Title: 莫烦/ 建造第一个神经网络/ Lesson2-区分类型（分类）
Main Author: Morvan Zhou
Editor: Shengjie Xiu
Time: 2019/3/21
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

# 分类问题：类型0和类型1，分别在(2,2)附近和(-2,2)附近

import torch
import matplotlib.pyplot as plt


# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2 * n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2 * n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
# FloatTensor = 32-bit floating 按维数0（行）拼接
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat(
    (y0, y1), 0).type(
        torch.LongTensor)    # LongTensor = 64-bit integer

# 画图
#plt.scatter(x.data.numpy(), y.data.numpy())
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()


# 构建网络
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = torch.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)  # 几个类别就几个 output

print(net)

# 训练网络
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 接着上面来
    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(torch.softmax(out, 1), 1)[1]  #此处softmax可以不做
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(
            x.data.numpy()[
                :, 0], x.data.numpy()[
                :, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(
            1.5, -4, 'Accuracy=%.2f' %
            accuracy, fontdict={
                'size': 20, 'color': 'red'})
        plt.pause(0.1)

    if t % 10 == 0:
        print(loss)


plt.ioff()  # 停止画图
plt.show()
