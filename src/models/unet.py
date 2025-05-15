import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net architecture for image-to-image prediction.
    Used for predicting crop yields based on environmental data.
    """
    def __init__(self, in_channels=15, out_channels=3):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)  # 1024 = 512 + 512 (skip connection)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # 512 = 256 + 256 (skip connection)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)  # 256 = 128 + 128 (skip connection)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)  # 128 = 64 + 64 (skip connection)
        
        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)
        
        # Decoder
        upconv4 = self.upconv4(bottleneck)
        concat4 = torch.cat((upconv4, enc4), dim=1)
        dec4 = self.dec4(concat4)
        
        upconv3 = self.upconv3(dec4)
        concat3 = torch.cat((upconv3, enc3), dim=1)
        dec3 = self.dec3(concat3)
        
        upconv2 = self.upconv2(dec3)
        concat2 = torch.cat((upconv2, enc2), dim=1)
        dec2 = self.dec2(concat2)
        
        upconv1 = self.upconv1(dec2)
        concat1 = torch.cat((upconv1, enc1), dim=1)
        dec1 = self.dec1(concat1)
        
        # Output
        out = self.out(dec1)
        out = self.sigmoid(out)
        
        return out