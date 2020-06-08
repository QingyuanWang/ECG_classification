from tqdm import tqdm
import torch
from util import MovAvg


class Trainer:
    def __init__(self, model, loader_train, loader_test, optimizer):
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.optimizer = optimizer

    def metric(self, preds, labels, loss, batch_size):
        preds_ = torch.argmax(preds, dim=-1)
        acc = (preds_ == labels).sum() / batch_size

        info = {
            'acc': acc,
            'loss': loss,
        }
        for k in info.keys():
            info[k] = info[k].item()
        return info

    def train_with_val(self, loss_fn, epochs, save_path=None, save_best_only=False, monitor_on=None):
        best_metric = 0
        for epoch in range(epochs):
            self.train(loss_fn, epoch, save_path)
            info = self.test(loss_fn, epoch)
            if save_path is not None:
                if not save_best_only:
                    torch.save(self.model.state_dict(), save_path)
                elif float(info[monitor_on]) > best_metric:
                    print(f'Save model at {monitor_on}={info[monitor_on]}')
                    best_metric = float(info[monitor_on])
                    torch.save(self.model.state_dict(), save_path)

    def _update_info(self, preds, labels, loss, info_avg):
        with torch.no_grad():
            info = self.metric(preds, labels, loss, float(labels.shape[0]))
            info_show = {}
            for k in info.keys():
                if k not in info_avg.keys():
                    info_avg[k] = MovAvg()
                info_avg[k].add(info[k])
                info_avg[k].add(info[k])
                info_show[k] = f'{info_avg[k].get():.4f}'
            return info_show

    def train(self, loss_fn, epoch=0, save_path=None):
        self.model.train()
        desc = f'Epoch #{epoch + 1}' if epoch != 0 else None
        with tqdm(total=len(self.loader_train), desc=desc) as progress_bar:
            info_avg = {}
            for batch_idx, (data, labels) in enumerate(self.loader_train):
                self.optimizer.zero_grad()
                preds = self.model(data)
                loss = loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                info_show = self._update_info(preds, labels, loss, info_avg)
                progress_bar.set_postfix(**info_show)
                progress_bar.update(1)
        return info_show

    def test(self, loss_fn, epoch=0):
        self.model.eval()
        desc = f'Epoch #{epoch + 1}' if epoch != 0 else None
        with torch.no_grad():
            with tqdm(total=len(self.loader_test), desc=desc) as progress_bar:
                info_avg = {}
                for batch_idx, (data, labels) in enumerate(self.loader_test):
                    preds = self.model(data)
                    loss = loss_fn(preds, labels)
                    info_show = self._update_info(preds, labels, loss, info_avg)
                    progress_bar.set_postfix(**info_show)
                    progress_bar.update(1)
        return info_show
