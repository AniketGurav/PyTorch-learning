"""
Title: 莫烦/ 高级神经网络结构/ Lesson1-CNN
Main Author: Morvan Zhou
Editor: Shengjie Xiu
Time: 2019/3/28
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim    # import for PyCharm autocomplete convenience
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# paramaters
LR = 0.01
BATCH_SIZE = 50
EPOCH = 1
DOWNLOAD_MNIST = False   # if have not downloaded, switch to True
GPU = True  # GPU and CPU switch

# prepare data
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x
    # H x W) and normalize in the range [0.0, 1.0]
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2)

test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False
)

test_x = torch.unsqueeze(
    test_data.data, dim=1).type(
        torch.FloatTensor)[
            :2000] / 255.
test_y = test_data.targets[:2000]


# build the network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.predict = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = self.predict(x)
        return x


cnn = CNN()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), LR, betas=(0.9, 0.999))

loss_record = []
test_acc_record = []


def forward_accuracy(data, label, net, device):
    if device != "cpu":
        data=data.to(device)
    output = net.forward(data)
    if device != "cpu":
        output = output.cpu()
    predict_label = torch.max(output, 1)[1].data.numpy()
    label = label.numpy()
    accuracy = sum(predict_label == label) / np.size(predict_label, 0)
    return accuracy


if __name__ == '__main__':

#1. GPU CPU switch
    if GPU == True and torch.cuda.is_available():
        device = torch.device("cuda:0")
        cnn = cnn.to(device)
    else:
        device = 'cpu'
    print('We are using', device)

#2. training
    for epoch in range(EPOCH):
        print('Epoch', epoch)
        for step, (data, label) in enumerate(train_loader):
            if step <= 300:

                if device != "cpu":
                    # data and label to GPU
                    data=data.to(device)
                    label=label.to(device)

                prediction = cnn.forward(data)
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if device != "cpu":
                    # loss back to CPU
                    loss=loss.cpu()

                loss_record.append(loss.data.numpy())

                if step % 10 == 9:
                    print('step: ', step + 1, 'error', loss.data.numpy())
                    test_acc = forward_accuracy(
                        test_x, test_y, cnn, device)
                    print('test accuracy: ', test_acc)
                    test_acc_record.append(test_acc)

    plt.subplot(211)
    plt.title('Training Loss')
    plt.plot(loss_record)
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.subplot(212)
    plt.title('Testing Accuracy')
    plt.plot(test_acc_record)
    plt.xlabel('10 Steps')
    plt.ylabel('test accuracy')
    plt.show()

