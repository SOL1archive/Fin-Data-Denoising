import torch
from torch.utils.data import Dataset

class SeqWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.window_size]
