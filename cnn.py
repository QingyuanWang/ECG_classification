from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# create cnn
class net(nn.Module):
    def __init__(self):
        # structure
        super(net, self).__init__()
        self.conv1 = nn.Conv1d(2, 128, 50, stride=3)
        self.conv2 = nn.Conv1d(128, 32, 7, stride=1)
        self.conv3 = nn.Conv1d(32, 32, 10, stride=1)
        self.conv4 = nn.Conv1d(32, 128, 5, stride=2)
        self.conv5 = nn.Conv1d(128, 256, 15, stride=1)
        self.conv6 = nn.Conv1d(256, 512, 5, stride=1)
        self.conv7 = nn.Conv1d(512, 128, 3, stride=1)
        self.dense1 = nn.Linear(1152, 512)
        self.dense2 = nn.Linear(512, 17)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(0.1)
        self.faltten = nn.Flatten()

    # forward propagation
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool1d(x, 2, stride=3)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool1d(x, 2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, 2, stride=2)

        x = F.relu(self.conv5(x))
        x = F.max_pool1d(x, 2, stride=2)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = self.faltten(x)
        x = self.dropout(F.relu(self.dense1(x)))
        output = self.dense2(x)

        return output