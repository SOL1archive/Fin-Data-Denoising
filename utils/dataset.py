from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data.values.astype('float32')  # convert data to numpy array of type float32
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]  # select the first three columns as input
        y = self.data[index]   # select the fourth column as output
        return x, y
