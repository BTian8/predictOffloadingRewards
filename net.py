import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

"""Dataloader and NN model for offloading reward estimation."""


class EdgeDetectionDataset(Dataset):
    def __init__(self, inputs, labels, transform=ToTensor(), target_transform=ToTensor()):
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        label = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        return x, label


class EdgeDetectionNet(nn.Module):
    def __init__(self, channels, kernels, pools, linear=[]):
        """
        Build a convolutional neural network to predict offloading reward using feature maps from the weak detector.
        :param channels: a list with the number of channel for each convolutional layer.
        :param kernels: a list with the kernel size for each convolutional layer.
        :param pools: a boolean list that specifies whether each convolutional layer should followed by a pooling layer.
        :param linear: an optional list that specifies the number of features in each linear (fully-connected) layer.
                       If an empty list is given, build a fully-convolutional network ends with global average pooling.
        """
        super(EdgeDetectionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_stacks, self.linear_stacks = [], []
        # Construct convolutional and linear stacks.
        for in_channel, out_channel, kernel_size, pool in zip(channels[:-1], channels[1:], kernels, pools):
            self.conv_stacks.append(self.conv_stack(in_channel, out_channel, kernel_size, pool))
        if len(linear) > 0:
            drops = [True] * (len(linear) - 1)
            drops[-1] = False
            for in_feature, out_feature, dropout in zip(linear[:-1], linear[1:], drops):
                self.linear_stacks.append(self.linear_stack(in_feature, out_feature, dropout))

    def conv_stack(self, in_channels, out_channels, kernel_size, pool):
        """
        Build a convolutional layer with relu activator, batch normalization, and (optional) max pooling.
        :param in_channels: number of channels in the input feature map.
        :param out_channels: number of channels in the output feature map.
        :param kernel_size: size of kernel in the convolutional layer.
        :param pool: if max pooling is applied.
        :return: the constructed convolutional stack.
        """
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                   self.relu,
                   nn.BatchNorm2d(out_channels)]
        if pool:
            modules.append(self.pool)
        conv = nn.Sequential(*modules)
        return conv

    def linear_stack(self, in_features, out_features, dropout=False):
        """
        Build a linear (fully-connected) layer with relu activator, and (optional) dropout.
        :param in_features: number of features in the input.
        :param out_features: number of channels in the output.
        :param dropout: whether dropout is applied.
        :return: the constructed linear stack.
        """
        modules = [nn.Linear(in_features, out_features), self.relu]
        if dropout:
            modules.append(nn.Dropout())
        linear = nn.Sequential(*modules)
        return linear

    def forward(self, x):
        """Forward function."""
        for conv in self.conv_stacks:
            x = conv(x)
        x = self.flatten(x)
        if len(self.linear_stacks) > 0:
            for linear in self.linear_stacks:
                x = linear(x)
        else:
            # Use global average pooling if no fully-connected layer is applied.
            x = torch.mean(x)
        return x
