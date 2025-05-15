import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class UrbanCropDataset(Dataset):
    """Dataset for urban-crop yield prediction models"""
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: Input data array with shape [N, C, H, W] or [N, T, H, W, C]
            y: Target data array with shape [N, C, H, W]
            transform: Optional transform to be applied to samples
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]
        
        if self.transform:
            x_sample = self.transform(x_sample)
            
        return x_sample, y_sample

class TemporalUrbanCropDataset(Dataset):
    """Dataset that reshapes data for LSTM-UNet models to handle temporal information"""
    
    def __init__(self, X, y, time_steps=23, transform=None):
        """
        Args:
            X: Input data with shape [N, C, H, W]
            y: Target data with shape [N, C, H, W]
            time_steps: Number of time steps to reshape the data into
            transform: Optional transform to be applied to samples
        """
        # Get dimensions
        N, C, H, W = X.shape
        
        # Reshape X to [N, time_steps, H, W, C/time_steps]
        # Assuming C is divisible by time_steps
        channels_per_step = C // time_steps
        X_reshaped = X.view(N, time_steps, channels_per_step, H, W)
        X_reshaped = X_reshaped.permute(0, 1, 3, 4, 2)  # [N, time_steps, H, W, channels_per_step]
        
        self.X = torch.tensor(X_reshaped, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]
        
        if self.transform:
            x_sample = self.transform(x_sample)
            
        return x_sample, y_sample