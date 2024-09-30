import torch
from torch.utils.data import Dataset
# Define a Dataset
class TorchDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.from_numpy(x.astype('float32'))
        self.y = torch.from_numpy(y.astype('float32'))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index].unsqueeze(0)