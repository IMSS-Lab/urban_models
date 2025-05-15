import torch
import torch.nn as nn

class BidirectionalHybrid(nn.Module):
    def __init__(self, input_shape=(23, 32, 32, 3), lstm_units=32):
        super().__init__()
        self.time_steps = input_shape[0]
        
        # Spatio-temporal processor
        self.lstm = nn.LSTM(
            input_size=32*32*3,
            hidden_size=lstm_units,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # U-Net components
        self.encoder = nn.Sequential(
            nn.Conv2d(lstm_units*2, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM processing
        x = x.view(batch_size, self.time_steps, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :].unsqueeze(-1).unsqueeze(-1)
        
        # U-Net processing
        x = x.expand(-1, -1, 32, 32)
        x = self.encoder(x)
        return self.decoder(x)