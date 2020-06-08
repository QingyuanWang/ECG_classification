import numpy as np
import torch


class MovAvg(object):
    def __init__(self, size: int = 100) -> None:
        super().__init__()
        self.size = size
        self.cache = []

    def add(self, x):
        """Add a scalar into :class:`MovAvg`. You can add ``torch.Tensor`` with
        only one element, a python scalar, or a list of python scalar.
        """
        if isinstance(x, torch.Tensor):
            x = x.item()
        self.cache.append(x)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self) -> float:
        """Get the average."""
        if len(self.cache) == 0:
            return 0
        return np.mean(self.cache)

    def mean(self) -> float:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> float:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0
        return np.std(self.cache)
