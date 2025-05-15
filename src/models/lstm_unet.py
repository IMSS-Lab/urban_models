import torch
import torch.nn as nn

class LSTMUNet(nn.Module):
    """
    Hybrid LSTM-UNet model for spatio-temporal prediction.
    Uses LSTM to process temporal data, then a U-Net architecture for spatial processing.
    """
    def __init__(self, input_shape=(23, 32, 32, 3), lstm_units=16, unet_filters=16):
        super(LSTMUNet, self).__init__()
        
        self.time_steps, self.height, self.width, self.channels = input_shape
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(
            input_size=self.height * self.width * self.channels,
            hidden_size=lstm_units,
            batch_first=True
        )
        self.dense = nn.Linear(lstm_units, self.height * self.width * lstm_units)
        
        # Simplified U-Net
        # Encoder
        self.conv1 = nn.Conv2d(lstm_units, unet_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.conv2 = nn.Conv2d(unet_filters, unet_filters*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Decoder
        self.upconv = nn.ConvTranspose2d(unet_filters*2, unet_filters, kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(unet_filters*2, unet_filters, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Output layer
        self.out = nn.Conv2d(unet_filters, 3, kernel_size=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for LSTM: [batch, time_steps, height, width, channels] -> [batch, time_steps, flattened]
        x_flat = x.view(batch_size, self.time_steps, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_flat)
        lstm_last = lstm_out[:, -1, :]  # Take the last time step output
        
        # Dense projection and reshape to spatial dimensions
        dense_out = self.dense(lstm_last)
        spatial = dense_out.view(batch_size, -1, self.height, self.width)
        
        # U-Net processing
        # Encoder
        conv1 = self.relu1(self.conv1(spatial))
        pool1 = self.pool1(conv1)
        
        # Bottleneck
        conv2 = self.relu2(self.conv2(pool1))
        
        # Decoder
        upconv = self.upconv(conv2)
        concat = torch.cat([upconv, conv1], dim=1)
        conv3 = self.relu3(self.conv3(concat))
        
        # Output
        out = self.out(conv3)
        
        return out


class BidirectionalLSTMUNet(nn.Module):
    """
    Bidirectional LSTM-UNet model for improved temporal information processing.
    """
    def __init__(self, input_shape=(23, 32, 32, 3), lstm_units=16, unet_filters=16):
        super(BidirectionalLSTMUNet, self).__init__()
        
        self.time_steps, self.height, self.width, self.channels = input_shape
        
        # Temporal processing with Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.height * self.width * self.channels,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True
        )
        self.dense = nn.Linear(lstm_units*2, self.height * self.width * lstm_units)  # *2 for bidirectional
        
        # Simplified U-Net
        # Encoder
        self.conv1 = nn.Conv2d(lstm_units, unet_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.conv2 = nn.Conv2d(unet_filters, unet_filters*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Decoder
        self.upconv = nn.ConvTranspose2d(unet_filters*2, unet_filters, kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(unet_filters*2, unet_filters, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Output layer
        self.out = nn.Conv2d(unet_filters, 3, kernel_size=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for LSTM: [batch, time_steps, height, width, channels] -> [batch, time_steps, flattened]
        x_flat = x.view(batch_size, self.time_steps, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_flat)
        lstm_last = lstm_out[:, -1, :]  # Take the last time step output
        
        # Dense projection and reshape to spatial dimensions
        dense_out = self.dense(lstm_last)
        spatial = dense_out.view(batch_size, -1, self.height, self.width)
        
        # U-Net processing
        # Encoder
        conv1 = self.relu1(self.conv1(spatial))
        pool1 = self.pool1(conv1)
        
        # Bottleneck
        conv2 = self.relu2(self.conv2(pool1))
        
        # Decoder
        upconv = self.upconv(conv2)
        concat = torch.cat([upconv, conv1], dim=1)
        conv3 = self.relu3(self.conv3(concat))
        
        # Output
        out = self.out(conv3)
        
        return out