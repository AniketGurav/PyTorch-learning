"""
Title: 莫烦/ 高级神经神经网络结构/ Lesson2-RNN-Classification
Main Author: Morvan Zhou
Editor: Shengjie Xiu
Time: 2019/3/28
Purpose: PyTorch learning
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

#############################################################################
# comment by Shengjie Xiu -- The understanding of LSTM for image in PyTorch
#
#
# This task uses LSTM to classify MNIST image (28*28)
# Method: We use LSTM to scan each row of image in each block
#
# training step:
# 1) train_loader of torch.Size([50, 1, 28, 28])  "from MINST dataset"
#
# 2) use .view(-1, 28, 28) -> input image batch of torch.Size([50, 28, 28])
#
# 3) First layer LSTM
#   contains a series of LSTM block
#
# 3.1) Inputs: input, (h_0, c_0):
# {1} input: train_loader of torch.Size([50, 28, 28])
# {2} (h_0, c_0): The LSTM hidden state & cell state in the past -> set to None
#
# 3.2) Network parameters:
# {1} input_size: The number of expected features in the input xt. In
#     this case in each LSTM block we input one row of image batch as xt,
#     and each row has 28 features (image width) -> set to 28
#     (still for the network)
# {2} hidden_size: The number of features in the hidden state ht. From 28
#     features we learn 64 -> set to 64
# {3} num_layers: Number of recurrent layers in one LSTM block. -> set to 1
# {4} batch_first: default input of size (seq_len, batch, input_size), but we
#     have (batch, seq_len, input_size) -> set to True
# {notice} seq_len(image height): Determines the number of blocks. Since the
#     LSTM is a iterable structure, the later weight is gained from the previous,
#     so seq_len is not required to define the layer. Means scalable.
#
# 3.3) Outputs: output, (h_n, c_n)
# {1} output: Record the output of each LSTM block.
#     Default output of size (seq_len, batch, num_directions * hidden_size)
#     but as "batch_first=True" and we just have 1 direction (not bidirectional),
#     we gain output of torch.Size([50, 28, 64]). Since we just need the output
#     of the last block, we use output[:, -1, :].
# {2} (h_n, c_n): Record only the hidden state and cell state of the last block.
#     In this case both of them are of torch.Size([1, 50, 64])
#
# 4) Second layer FC
#    Use the output[:, -1, :] of LSTM as input, 64 features map to 10 classes.
#############################################################################

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim    # import for PyCharm autocomplete convenience
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 50
IMAGE_WEIDTH = 28
EPOCH = 1
LR = 0.01
DOWNLOAD_MNIST = False
GPU = True

# prepare data
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
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
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=IMAGE_WEIDTH,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.predict = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_out, c_out) = self.rnn(x, None)
        out = self.predict(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), LR)

loss_record = []
test_acc_record = []


def forward_accuracy(data, label, net, device):
    data = data.view(-1, 28, 28)
    if device != "cpu":
        data = data.to(device)
    output = net.forward(data)
    if device != "cpu":
        output = output.cpu()
    predict_label = torch.max(output, 1)[1].data.numpy()
    label = label.numpy()
    accuracy = sum(predict_label == label) / np.size(predict_label, 0)
    return accuracy


if __name__ == '__main__':

    # 1. GPU CPU switch
    if GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        rnn = rnn.to(device)
    else:
        device = 'cpu'
    print('We are using', device)

    # 2. training
    for epoch in range(EPOCH):
        print('Epoch', epoch)
        for step, (data, label) in enumerate(train_loader):

            data = data.view(-1, 28, 28)

            if device != "cpu":
                # data and label to GPU
                data = data.to(device)
                label = label.to(device)

            prediction = rnn.forward(data)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if device != "cpu":
                # loss back to CPU
                loss = loss.cpu()

            loss_record.append(loss.data.numpy())

            if step % 10 == 9:
                print('step: ', step + 1, 'error', loss.data.numpy())
                test_acc = forward_accuracy(
                    test_x, test_y, rnn, device)
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
