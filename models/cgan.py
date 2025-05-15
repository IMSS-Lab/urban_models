import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, layer_sizes, z_size, img_size, num_classes):
        super().__init__()
        self.z_size = z_size
        self.img_size = img_size
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        layers = []
        input_size = z_size + num_classes
        for i, size in enumerate(layer_sizes):
            layers += [
                nn.Linear(input_size, size),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            input_size = size
        
        layers.append(nn.Linear(layer_sizes[-1], img_size * img_size))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z, labels):
        z = z.view(-1, self.z_size)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, self.img_size, self.img_size)

class Discriminator(nn.Module):
    def __init__(self, layer_sizes, img_size, num_classes):
        super().__init__()
        self.img_size = img_size
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        layers = []
        input_size = img_size * img_size + num_classes
        for i, size in enumerate(layer_sizes):
            layers += [
                nn.Linear(input_size, size),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ]
            input_size = size
        
        layers += [
            nn.Linear(layer_sizes[-1], 1),
            nn.Sigmoid()
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, labels):
        x = x.view(-1, self.img_size * self.img_size)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        return self.model(x).squeeze()