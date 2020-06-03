# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import wfdb
import DataProcess
from cnn import net
import numpy as np
import torch.utils.data as Data
import torch
import os
import json



# %% Data Processing
# DataProcess.process_data()

# %%

torch.manual_seed(1)
model = net()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

acc_max = -1
bestepoch = -1

log_dir = "./log_dir/"
checkpoint_dir = "./checkpoint_dir/"
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)

data = np.load('./data.npz')
length = data['X'].shape[0]
split = 0.9
l_train = int(length * split)
X_train = torch.tensor(data['X'][0:l_train], dtype=torch.float).permute([0, 2, 1])
Y_train = torch.tensor(data['Y'][0:l_train], dtype=torch.long)

l_test = int(length * (1-split))
X_test = torch.tensor(data['X'][0:l_test], dtype=torch.float).permute([0, 2, 1])
Y_test = torch.tensor(data['Y'][0:l_test], dtype=torch.long)

torch_dataset_train = Data.TensorDataset(X_train, Y_train)
torch_dataset_test = Data.TensorDataset(X_test, Y_test)

loader_train = Data.DataLoader(
    dataset=torch_dataset_train,
    batch_size=100,
    shuffle=True,
    num_workers=2,
)

loader_test = Data.DataLoader(
    dataset=torch_dataset_test,
    batch_size=100,
    shuffle=False,
    num_workers=2,
)

num_epochs = 5
for epoch in range(num_epochs):
    acc, loss = model.train_on_dataset(loader_train, optimizer)
    if acc >= acc_max:
        acc_max = acc
        bestepoch = epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir,"checkpoint_CNN"))
        bestlog ={
            "loss": float(loss),
            "acc": float(acc),
            "best_epoch": int(epoch),
            "lr": 3e-4,
            "batch_size":100
        }  

print('Best validation Epoch is {}, Weights and Bias saved in checkpoints.'.format(int(bestepoch)))
with open(os.path.join(log_dir,"BestCNN_epoch{}.json".format(bestepoch)), 'w') as f:
    json.dump(bestlog, f)
