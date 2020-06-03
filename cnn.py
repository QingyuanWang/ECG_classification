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

    def train_on_dataset(self, loader, optimizer):
        self.train()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cost = nn.CrossEntropyLoss()
        with tqdm(total=len(loader)) as progress_bar:
            for batch_idx, (data, label) in enumerate(loader):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = self(data)
                loss = cost(output, label)

                preds = torch.argmax(output, axis=-1)
                acc = (preds == label).sum() /  float(label.shape[0])
                info = {
                    'loss': loss.item(),
                    'acc': acc.item()
                }
                progress_bar.set_postfix(**info)
                # back propagation
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
        return acc, loss

    # validation, inlcudes evaluation on validation set
    def test_on_dataset(self, test_loader, checkpoint=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if checkpoint is not None:
            model_state = torch.load(checkpoint)
            self.load_state_dict(model_state)
        # Need this line for things like dropout etc.
        # evalution
        self.eval()
        preds = []
        targets = []
        cost = nn.CrossEntropyLoss()
        losses = []

        with torch.no_grad():
            # calculate a series of variables for evaluation
            for batch_idx, (data, label) in enumerate(test_loader):
                # original data
                data = data.to(device)
                # labels (0,1)
                target = label.clone()
                # predicted labels
                output = self(data)
                preds.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
                output = output.to(device)
                label = label.to(device)
                # loss
                losses.append(cost(output, label))
        # average loss value
        loss = torch.mean(torch.stack(losses))
        # reshape predicted labels
        preds = np.argmax(np.concatenate(preds), axis=1)
        # reshape original labels
        targets = np.concatenate(targets)
        # accurancy
        acc = accuracy_score(targets, preds)

        return acc, loss
