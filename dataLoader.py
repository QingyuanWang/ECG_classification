import torch
import torch.utils.data as Data
import numpy as np


class DataLoader:
    def __init__(self, X, Y, train_val_split=[0.7, 0.15, 0.15], shuffle=True, batch_size=100, device='cpu'):
        if shuffle:
            p = np.random.permutation(range(len(X)))
            X, Y = X[p], Y[p]
        self.loader_list = []
        assert len(train_val_split) > 1
        end_index = 0
        for i in range(len(train_val_split)):
            start_index = end_index
            end_index += int(len(X) * train_val_split[i])
            X_tensor = torch.tensor(X[start_index:end_index], dtype=torch.float, device=device).permute([0, 2, 1])
            Y_tensor = torch.tensor(Y[start_index:end_index], dtype=torch.long, device=device)
            dataset = Data.TensorDataset(X_tensor, Y_tensor)
            self.loader_list.append(Data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
            ))

    def get_loader(self):
        return self.loader_list
