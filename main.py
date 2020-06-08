from cnn import net
import numpy as np
import torch
import os
from trainer import Trainer
from dataLoader import DataLoader

torch.manual_seed(1)

checkpoint_dir = "./checkpoint_dir/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

data = np.load('./data.npz')

model = net()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

dataLoader = DataLoader(data['X'], data['Y'], train_val_split=0.9)
loader_train, loader_test = dataLoader.get_loader()
trainer = Trainer(model, loader_train, loader_test, optimizer)
loss_fn = torch.nn.functional.cross_entropy
trainer.train_with_val(loss_fn,
                       epochs=2,
                       save_path=checkpoint_dir + 'model.pth',
                       save_best_only=True,
                       monitor_on='acc')
