import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)

        return out


class Net1(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, stride):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv3d(n_input, n_hidden, stride=stride)
        # self.dense1 = nn.Linear(n_hidden, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = F.max_pool3d(input)
        # out = out.view(data size)
        out = F.relu(self.dense1(out))
        out = self.dense2(out)
        return out
