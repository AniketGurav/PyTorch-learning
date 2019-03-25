"""
Title: PyTorch/ Get Started/ a 60-min Blitz/ Training a Classifier
Main Author: PyTorch
Editor: Shengjie Xiu
Time: 2019/3/25
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# We will do the following steps in order:
#
# 1. Load and normalizing the CIFAR10 training and test datasets using
#    ``torchvision``
# 2. Define a Convolutional Neural Network
# 3. Define a loss function
# 4. Train the network on the training data
# 5. Test the network on the test data

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 1)load traning set
# using CIFAR10 from pytorch torchvision
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


# 2)generate training batch
# torch.utils.data.DataLoade--Parameters:
# dataset (Dataset) – dataset from which to load the data.
# batch_size (int, optional) – how many samples per batch to load (default: 1).
# shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
# num_workers (int, optional) – how many subprocesses to use for data
# loading. 0 means that the data will be loaded in the main process.
# (default: 0)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


# 3)load testing dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


# 4)testing batch
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# define class
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 5)build neural network in a faster way
# net = torch.nn.Sequential(
#     torch.nn.Conv2d(3,6,5),
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(2),
#     torch.nn.Conv2d(6,16,5),
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(2),
#     torch.nn.Linear(16*5*5,120),
#     torch.nn.Linear(120, 84),
#     torch.nn.Linear(84, 10)
# )

#5)build neural network in a traditional way
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)    # the size -1 is inferred from other
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
To build a network with "view", traditional method using "class Net" is preferable
'''

net = Net()

# 6)define loss function and optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 7)train the network
if __name__ == '__main__':  # to cope with the problem of multi-process, we use main function
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


'''
Run:result

Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
[1,  2000] loss: 2.196
[1,  4000] loss: 1.844
[1,  6000] loss: 1.674
[1,  8000] loss: 1.576
[1, 10000] loss: 1.519
[1, 12000] loss: 1.482
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
[2,  2000] loss: 1.406
[2,  4000] loss: 1.396
[2,  6000] loss: 1.356
[2,  8000] loss: 1.336
[2, 10000] loss: 1.317
[2, 12000] loss: 1.309
Finished Training
'''