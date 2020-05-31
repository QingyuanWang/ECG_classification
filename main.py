# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import wfdb
import DataProcess
from cnn import net
import numpy as np
import torch.utils.data as Data
import torch

# %%
# DataProcess.process_data()

# %%
torch.manual_seed(1)
model = net()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

data = np.load('./data.npz')
length = data['X'].shape[0]
split = 0.9
l_train = int(length * split)
X_train = torch.tensor(data['X'][0:l_train], dtype=torch.float).permute([0, 2, 1])
Y_train = torch.tensor(data['Y'][0:l_train], dtype=torch.long)

torch_dataset = Data.TensorDataset(X_train, Y_train)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=100,
    shuffle=True,  # 每次训练打乱数据， 默认为False
    num_workers=2,  # 使用多进行程读取数据， 默认0，为不使用多进程
)
for i in range(epoch):
    model.train_on_dataset(loader, optimizer)

# %%
