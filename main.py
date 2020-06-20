from cnn import net
import numpy as np
import torch
import os
from trainer import Trainer
from dataLoader import DataLoader
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_dir/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset_path', type=str, default='./data.npz')
    args = parser.parse_known_args()[0]
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

args = get_args()
setup_seed(args.seed)
device = args.device
checkpoint_dir = args.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

data = np.load(args.dataset_path)

model = net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

dataLoader = DataLoader(data['X'],
                        data['Y'],
                        train_val_split=[0.7, 0.15, 0.15],
                        batch_size=args.batch_size,
                        device=device)
loader_train, loader_val, loader_test = dataLoader.get_loader()
trainer = Trainer(model, optimizer)
loss_fn = torch.nn.functional.cross_entropy
trainer.train_with_val(loss_fn,
                       loader_train=loader_train,
                       loader_val=loader_val,
                       epochs=args.epochs,
                       save_path=checkpoint_dir + 'model.pth',
                       save_best_only=True,
                       monitor_on='acc')
trainer.test(loader_test, loss_fn, info='Test ')
