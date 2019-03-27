"""
Title: 莫烦/ 建造第一个神经网络/ Lesson6-优化器
Main Author: Morvan Zhou
Editor: Shengjie Xiu
Time: 2019/3/27
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 20


# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


# pack the data into dataset
torch_dataset=Data.TensorDataset(x,y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# build neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net_SGD=Net()
net_Momentum=Net()
net_RMSprop=Net()
net_Adam=Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]


# different optimizers
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]


# loss function
loss_func=torch.nn.MSELoss()
losses_group=[[],[],[],[]]

if __name__ == '__main__':
    # train
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (b_x, b_y) in enumerate(loader):
            # update each network with specific optimizer
            for net, opt, l in zip(nets, optimizers, losses_group):
                predict=net(b_x)
                loss=loss_func(predict,b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l.append(loss.data.numpy())

    '''
    about Zip: pack iterable factors into several turples
    >>>a = [1,2,3]
    >>>b = [4,5,6]
    >>>zipped = zip(a,b)
    >>> list(zipped) 
    [(1, 4), (2, 5), (3, 6)]
    '''

    label = ["SGD", "Momentum", "RMSprop", "Adam"]
    for i, l in enumerate(losses_group):
        plt.plot(l, label=label[i])
    plt.legend(loc='best')  # to show the label
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()



