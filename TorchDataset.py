import torch
from torch.utils.data import Dataset
# Define a Dataset
class TorchDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index+len(self.x)-1]