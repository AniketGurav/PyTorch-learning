"""
Title: PyTorch/ Get Started/ a 60-min Blitz/ OPTIONAL: DATA PARALLELISM
Main Author: Morvan Zhou
Editor: Shengjie Xiu
Time: 2019/3/27
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

# main code is from "MorvanZhou_2_NeuralNetwork_6_Optimizer.py"

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

LR = 0.01
BATCH_SIZE = 32
EPOCH = 3


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
losses_group1=[[],[],[],[]]
losses_group2=[[],[],[],[]]

if __name__ == '__main__':

    if torch.cuda.is_available():
        # train on GPU
        # 1) assign device
        device = torch.device("cuda:0")
        print('We are using', device)

        time_start_GPU = time.time()
        for epoch in range(EPOCH):
            print('Epoch: ', epoch)
            for step, (b_x, b_y) in enumerate(loader):

                # 2) move data(type tensor) to GPU as new variable
                b_x1=b_x.to(device)
                b_y1=b_y.to(device)

                # update each network with specific optimizer
                for net, opt, l in zip(nets, optimizers, losses_group1):

                    # 3) move network to  GPU
                    net1 = net.to(device)

                    predict=net1(b_x1)
                    loss=loss_func(predict,b_y1)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    #) 4) move result back to CPU for easier analysis
                    loss.data = loss.data.cpu()

                    l.append(loss.data.numpy())

        time_end_GPU = time.time()
        print('GPU time', time_end_GPU - time_start_GPU)

        label = ["SGD", "Momentum", "RMSprop", "Adam"]
        plt.subplot(211)
        for i, l in enumerate(losses_group1):
            plt.plot(l, label=label[i])
        plt.legend(loc='best')  # to show the label
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.ylim((0, 0.2))
        plt.show()


    # traditional method on CPU
    device = torch.device("cpu")
    print('We are using CPU')

    time_start_CPU = time.time()
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (b_x, b_y) in enumerate(loader):

            b_x2 = b_x.to(device)
            b_y2 = b_y.to(device)

            # update each network with specific optimizer
            for net, opt, l in zip(nets, optimizers, losses_group2):

                net2 = net.to(device)

                predict=net2(b_x2)
                loss=loss_func(predict,b_y2)
                opt.zero_grad()
                loss.backward()
                opt.step()

                l.append(loss.data.numpy())

    time_end_CPU = time.time()
    print('CPU time', time_end_CPU - time_start_CPU)

    label = ["SGD", "Momentum", "RMSprop", "Adam"]
    plt.subplot(212)
    for i, l in enumerate(losses_group2):
        plt.plot(l, label=label[i])
    plt.legend(loc='best')  # to show the label
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()



