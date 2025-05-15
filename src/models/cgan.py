import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    """
    Generator network for the Conditional GAN.
    """
    def __init__(self, z_size, class_num, generator_layer_size, img_size):
        super(Generator, self).__init__()
        
        self.z_size = z_size
        self.img_size = img_size
        
        self.label_emb = nn.Embedding(class_num, class_num)
        
        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], self.img_size * self.img_size),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        # Reshape z
        z = z.view(-1, self.z_size)
        
        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        
        # Concat image & label
        x = torch.cat([z, c], 1)
        
        # Generator out
        out = self.model(x)
        
        return out.view(-1, 1, self.img_size, self.img_size)  # Add channel dimension

class Discriminator(nn.Module):
    """
    Discriminator network for the Conditional GAN.
    """
    def __init__(self, img_size, class_num, discriminator_layer_size):
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        self.label_emb = nn.Embedding(class_num, class_num)
        
        self.model = nn.Sequential(
            nn.Linear(self.img_size * self.img_size + class_num, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        # Reshape fake image
        x = x.view(-1, self.img_size * self.img_size)
        
        # One-hot vector to embedding vector
        c = self.label_emb(labels)
        
        # Concat image & label
        x = torch.cat([x, c], 1)
        
        # Discriminator out
        out = self.model(x)
        
        return out.squeeze()

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, z_size, class_num, device):
    # Zero the gradients
    g_optimizer.zero_grad()
    
    # Generate random noise and labels
    z = torch.randn(batch_size, z_size).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, class_num, batch_size)).to(device)
    
    # Generate fake images
    fake_images = generator(z, fake_labels)
    
    # Get discriminator predictions on fake images
    validity = discriminator(fake_images, fake_labels)
    
    # Calculate loss - we want the discriminator to think these are real
    g_loss = criterion(validity, torch.ones(batch_size).to(device))
    
    # Backward pass and optimize
    g_loss.backward()
    g_optimizer.step()
    
    return g_loss.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, 
                             real_images, real_labels, z_size, class_num, device):
    # Zero the gradients
    d_optimizer.zero_grad()
    
    # Real images
    real_validity = discriminator(real_images, real_labels)
    real_loss = criterion(real_validity, torch.ones(batch_size).to(device))
    
    # Fake images
    z = torch.randn(batch_size, z_size).to(device)
    fake_labels = torch.LongTensor(np.random.randint(0, class_num, batch_size)).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images.detach(), fake_labels)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size).to(device))
    
    # Total discriminator loss
    d_loss = real_loss + fake_loss
    
    # Backward pass and optimize
    d_loss.backward()
    d_optimizer.step()
    
    return d_loss.item()