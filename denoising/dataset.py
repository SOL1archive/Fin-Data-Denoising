import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

class SeqWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        #self.pad_value = data.mean()
        self.pad_value = 0
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.window_size]
        
        return pad(seq, self.window_size, value=self.pad_value)
