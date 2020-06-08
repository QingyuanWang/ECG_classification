from tqdm import tqdm
import torch
from util import MovAvg


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
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

    def train_with_val(self,
                       loss_fn,
                       loader_train,
                       loader_val,
                       epochs,
                       save_path=None,
                       save_best_only=False,
                       monitor_on=None):
        best_metric = 0
        for epoch in range(epochs):
            self.train(loader_train, loss_fn, epoch, 'Train ', save_path)
            info = self.test(loader_val, loss_fn, epoch, 'Val ')
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

    def train(self, dataloader, loss_fn, epoch=-1, info='', save_path=None):
        self.model.train()
        desc = f'{info}Epoch #{epoch + 1}'
        with tqdm(total=len(dataloader), desc=desc) as progress_bar:
            info_avg = {}
            for batch_idx, (data, labels) in enumerate(dataloader):
                self.optimizer.zero_grad()
                preds = self.model(data)
                loss = loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                info_show = self._update_info(preds, labels, loss, info_avg)
                progress_bar.set_postfix(**info_show)
                progress_bar.update(1)
        return info_show

    def test(self, dataloader, loss_fn, epoch=-1, info=''):
        self.model.eval()
        desc = f'{info}Epoch #{epoch + 1}'
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc=desc) as progress_bar:
                info_avg = {}
                for batch_idx, (data, labels) in enumerate(dataloader):
                    preds = self.model(data)
                    loss = loss_fn(preds, labels)
                    info_show = self._update_info(preds, labels, loss, info_avg)
                    progress_bar.set_postfix(**info_show)
                    progress_bar.update(1)
        return info_show
