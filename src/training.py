import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os

class ModelTrainer:
    """Base trainer class for urban models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # Default settings - override in specific implementations
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {'mae': []}
        self.val_metrics = {'mae': []}
        
    def compile(self, optimizer='adam', learning_rate=0.001, criterion='mse'):
        """Configure the model training parameters"""
        # Set loss function
        if criterion == 'mse':
            self.criterion = nn.MSELoss()
        elif criterion == 'mae':
            self.criterion = nn.L1Loss()
        elif criterion == 'bce':
            self.criterion = nn.BCELoss()
        elif isinstance(criterion, nn.Module):
            self.criterion = criterion
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")
        
        # Set optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        return self
    
    def calculate_metrics(self, y_pred, y_true):
        """Calculate additional metrics for model evaluation"""
        # Mean Absolute Error
        mae = nn.L1Loss()(y_pred, y_true).item()
        
        return {'mae': mae}
    
    def train_step(self, x, y):
        """Perform one training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward pass
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        # Calculate additional metrics
        metrics = self.calculate_metrics(y_pred, y)
        
        return loss.item(), metrics
    
    def val_step(self, x, y):
        """Perform one validation step"""
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            
            # Calculate additional metrics
            metrics = self.calculate_metrics(y_pred, y)
        
        return loss.item(), metrics
    
    def fit(self, train_loader, val_loader=None, epochs=10, callbacks=None):
        """Train the model"""
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {'mae': []}
        self.val_metrics = {'mae': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            epoch_train_metrics = {'mae': 0}
            
            for batch_idx, (x, y) in enumerate(train_loader):
                batch_loss, batch_metrics = self.train_step(x, y)
                epoch_train_loss += batch_loss
                for metric in batch_metrics:
                    epoch_train_metrics[metric] += batch_metrics[metric]
            
            # Calculate average loss and metrics
            epoch_train_loss /= len(train_loader)
            for metric in epoch_train_metrics:
                epoch_train_metrics[metric] /= len(train_loader)
            
            self.train_losses.append(epoch_train_loss)
            for metric in self.train_metrics:
                self.train_metrics[metric].append(epoch_train_metrics[metric])
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                epoch_val_loss = 0
                epoch_val_metrics = {'mae': 0}
                
                with torch.no_grad():
                    for batch_idx, (x, y) in enumerate(val_loader):
                        batch_loss, batch_metrics = self.val_step(x, y)
                        epoch_val_loss += batch_loss
                        for metric in batch_metrics:
                            epoch_val_metrics[metric] += batch_metrics[metric]
                
                # Calculate average loss and metrics
                epoch_val_loss /= len(val_loader)
                for metric in epoch_val_metrics:
                    epoch_val_metrics[metric] /= len(val_loader)
                
                self.val_losses.append(epoch_val_loss)
                for metric in self.val_metrics:
                    self.val_metrics[metric].append(epoch_val_metrics[metric])
                
                # Print progress
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, '
                      f'Val Loss: {epoch_val_loss:.4f}, Train MAE: {epoch_train_metrics["mae"]:.4f}, '
                      f'Val MAE: {epoch_val_metrics["mae"]:.4f}')
            else:
                # Print progress without validation
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, '
                      f'Train MAE: {epoch_train_metrics["mae"]:.4f}')
            
            # Execute callbacks if provided
            if callbacks:
                for callback in callbacks:
                    callback(epoch, self.model, self.optimizer, 
                            epoch_train_loss, epoch_val_loss if val_loader else None,
                            epoch_train_metrics, epoch_val_metrics if val_loader else None)
        
        # Return history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_mae': self.train_metrics['mae'],
            'val_mae': self.val_metrics['mae'] if val_loader else None
        }
        
        return history
    
    def evaluate(self, test_loader):
        """Evaluate the model on the test set"""
        self.model.eval()
        test_loss = 0
        test_metrics = {'mae': 0}
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                loss, metrics = self.val_step(x, y)
                test_loss += loss
                for metric in metrics:
                    test_metrics[metric] += metrics[metric]
        
        # Calculate average loss and metrics
        test_loss /= len(test_loader)
        for metric in test_metrics:
            test_metrics[metric] /= len(test_loader)
        
        return test_loss, test_metrics
    
    def save_model(self, save_path):
        """Save the model to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        """Load the model from disk"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        
        print(f"Model loaded from {load_path}")
    
    def plot_history(self, save_path=None):
        """Plot the training and validation loss and metrics"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # MAE plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_metrics['mae'], label='Train MAE')
        if self.val_metrics['mae']:
            plt.plot(self.val_metrics['mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()

class LSTMUNetTrainer(ModelTrainer):
    """Specific trainer for LSTM-UNet models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model, device)
    
    def train_step(self, x, y):
        """Modified train step for LSTM-UNet models"""
        self.model.train()
        self.optimizer.zero_grad()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Forward pass
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        # Calculate additional metrics
        metrics = self.calculate_metrics(y_pred, y)
        
        return loss.item(), metrics

class CGANTrainer:
    """Trainer for Conditional GAN models"""
    
    def __init__(self, generator, discriminator, 
                 z_size=100, class_num=10, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.generator = generator
        self.discriminator = discriminator
        self.z_size = z_size
        self.class_num = class_num
        self.device = device
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Default settings
        self.criterion = nn.BCELoss()
        self.g_optimizer = None
        self.d_optimizer = None
        
        self.g_losses = []
        self.d_losses = []
    
    def compile(self, optimizer='adam', learning_rate=0.0001):
        """Configure the training parameters"""
        # Set optimizers
        if optimizer == 'adam':
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
            self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            self.g_optimizer = optim.SGD(self.generator.parameters(), lr=learning_rate)
            self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=learning_rate)
        elif optimizer == 'rmsprop':
            self.g_optimizer = optim.RMSprop(self.generator.parameters(), lr=learning_rate)
            self.d_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        return self
    
    def generator_train_step(self, batch_size):
        """Train the generator"""
        # Zero gradients
        self.g_optimizer.zero_grad()
        
        # Generate random noise and labels
        z = torch.randn(batch_size, self.z_size).to(self.device)
        fake_labels = torch.LongTensor(np.random.randint(0, self.class_num, batch_size)).to(self.device)
        
        # Generate fake images
        fake_images = self.generator(z, fake_labels)
        
        # Calculate the discriminator's predictions on the fake images
        validity = self.discriminator(fake_images, fake_labels)
        
        # Generator aims to make the discriminator think these are real
        g_loss = self.criterion(validity, torch.ones(batch_size).to(self.device))
        
        # Backpropagate and optimize
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def discriminator_train_step(self, real_images, real_labels, batch_size):
        """Train the discriminator"""
        # Zero gradients
        self.d_optimizer.zero_grad()
        
        # Get real batch
        real_images = real_images.to(self.device)
        real_labels = real_labels.to(self.device)
        
        # Train on real batch
        real_validity = self.discriminator(real_images, real_labels)
        real_loss = self.criterion(real_validity, torch.ones(batch_size).to(self.device))
        
        # Generate fake batch
        z = torch.randn(batch_size, self.z_size).to(self.device)
        fake_labels = torch.LongTensor(np.random.randint(0, self.class_num, batch_size)).to(self.device)
        fake_images = self.generator(z, fake_labels)
        
        # Train on fake batch
        fake_validity = self.discriminator(fake_images.detach(), fake_labels)
        fake_loss = self.criterion(fake_validity, torch.zeros(batch_size).to(self.device))
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        
        # Backpropagate and optimize
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def fit(self, train_loader, epochs=10, sample_interval=10, save_dir=None):
        """Train the CGAN"""
        self.g_losses = []
        self.d_losses = []
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_idx, (real_images, real_labels) in enumerate(train_loader):
                batch_size = real_images.size(0)
                
                # Train discriminator
                d_loss = self.discriminator_train_step(real_images, real_labels, batch_size)
                epoch_d_loss += d_loss
                
                # Train generator
                g_loss = self.generator_train_step(batch_size)
                epoch_g_loss += g_loss
            
            # Calculate average losses
            epoch_g_loss /= len(train_loader)
            epoch_d_loss /= len(train_loader)
            
            self.g_losses.append(epoch_g_loss)
            self.d_losses.append(epoch_d_loss)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}, G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}')
            
            # Save samples periodically
            if save_dir and epoch % sample_interval == 0:
                self.save_samples(epoch, save_dir)
        
        # Return history
        history = {
            'g_loss': self.g_losses,
            'd_loss': self.d_losses
        }
        
        return history
    
    def save_samples(self, epoch, save_dir):
        """Generate and save sample images"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate sample images for each class
        self.generator.eval()
        
        with torch.no_grad():
            # Create a grid of samples, one per class
            z = torch.randn(self.class_num, self.z_size).to(self.device)
            labels = torch.LongTensor(np.arange(self.class_num)).to(self.device)
            
            # Generate images
            samples = self.generator(z, labels)
            samples = samples.cpu().detach()
            
            # Create grid
            fig, axs = plt.subplots(1, self.class_num, figsize=(15, 2))
            for i in range(self.class_num):
                img = samples[i].squeeze().numpy()
                axs[i].imshow(img, cmap='gray')
                axs[i].set_title(f'Class {i}')
                axs[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
            plt.close()
    
    def save_model(self, save_dir):
        """Save the generator and discriminator models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save generator
        torch.save({
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.g_optimizer.state_dict(),
            'losses': self.g_losses
        }, os.path.join(save_dir, 'generator.pth'))
        
        # Save discriminator
        torch.save({
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.d_optimizer.state_dict(),
            'losses': self.d_losses
        }, os.path.join(save_dir, 'discriminator.pth'))
        
        print(f"Models saved to {save_dir}")
    
    def load_model(self, load_dir):
        """Load generator and discriminator models"""
        # Load generator
        g_checkpoint = torch.load(os.path.join(load_dir, 'generator.pth'), map_location=self.device)
        self.generator.load_state_dict(g_checkpoint['model_state_dict'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
        self.g_losses = g_checkpoint['losses']
        
        # Load discriminator
        d_checkpoint = torch.load(os.path.join(load_dir, 'discriminator.pth'), map_location=self.device)
        self.discriminator.load_state_dict(d_checkpoint['model_state_dict'])
        self.d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])
        self.d_losses = d_checkpoint['losses']
        
        print(f"Models loaded from {load_dir}")
    
    def plot_history(self, save_path=None):
        """Plot the generator and discriminator losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.title('GAN Losses')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()

# Useful callbacks
def model_checkpoint(filepath, monitor='val_loss', save_best_only=True):
    """Callback to save model when a metric improves"""
    best_val = float('inf') if 'loss' in monitor else float('-inf')
    
    def callback(epoch, model, optimizer, train_loss, val_loss, train_metrics, val_metrics):
        nonlocal best_val
        
        # Get the value to monitor
        if monitor == 'val_loss':
            current_val = val_loss
        elif monitor == 'train_loss':
            current_val = train_loss
        elif monitor == 'val_mae':
            current_val = val_metrics['mae']
        elif monitor == 'train_mae':
            current_val = train_metrics['mae']
        else:
            raise ValueError(f"Unsupported monitor: {monitor}")
        
        # Save if improved
        if (('loss' in monitor and current_val < best_val) or 
            ('loss' not in monitor and current_val > best_val) or 
            not save_best_only):
            
            # Update best value
            if save_best_only:
                best_val = current_val
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, filepath)
            
            print(f"Model saved to {filepath} (epoch {epoch+1}, {monitor}: {current_val:.4f})")
    
    return callback

def early_stopping(patience=10, monitor='val_loss'):
    """Callback to stop training when a metric stops improving"""
    best_val = float('inf') if 'loss' in monitor else float('-inf')
    wait = 0
    
    def callback(epoch, model, optimizer, train_loss, val_loss, train_metrics, val_metrics):
        nonlocal best_val, wait
        
        # Get the value to monitor
        if monitor == 'val_loss':
            current_val = val_loss
        elif monitor == 'train_loss':
            current_val = train_loss
        elif monitor == 'val_mae':
            current_val = val_metrics['mae']
        elif monitor == 'train_mae':
            current_val = train_metrics['mae']
        else:
            raise ValueError(f"Unsupported monitor: {monitor}")
        
        # Check if improved
        if (('loss' in monitor and current_val < best_val) or 
            ('loss' not in monitor and current_val > best_val)):
            best_val = current_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                raise KeyboardInterrupt("Early stopping triggered")
    
    return callback