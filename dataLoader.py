import torch
import torch.utils.data as Data


class DataLoader:
    def __init__(self, X, Y, train_val_split=0.9):
        train_len = int(X.shape[0] * train_val_split)
        X_train = torch.tensor(X[0:train_len], dtype=torch.float).permute([0, 2, 1])
        Y_train = torch.tensor(Y[0:train_len], dtype=torch.long)

        X_test = torch.tensor(X[train_len:], dtype=torch.float).permute([0, 2, 1])
        Y_test = torch.tensor(Y[train_len:], dtype=torch.long)

        torch_dataset_train = Data.TensorDataset(X_train, Y_train)
        torch_dataset_test = Data.TensorDataset(X_test, Y_test)

        self.loader_train = Data.DataLoader(
            dataset=torch_dataset_train,
            batch_size=100,
            shuffle=True,
        )

        self.loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=100,
            shuffle=False,
        )

    def get_loader(self):
        return self.loader_train, self.loader_test
