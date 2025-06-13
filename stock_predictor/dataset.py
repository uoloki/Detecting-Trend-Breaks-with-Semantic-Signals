# stock_predictor/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader

class PriceSequenceDataset(Dataset):
    """PyTorch Dataset for price sequences."""
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i].squeeze()

def get_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    """Creates train and test dataloaders."""
    train_ds = PriceSequenceDataset(X_train, y_train)
    test_ds = PriceSequenceDataset(X_test, y_test)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, test_dl