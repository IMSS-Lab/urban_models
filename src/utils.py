import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get the device to use for training"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_prediction(model, data_loader, num_samples=5, save_path=None):
    """Visualize model predictions side by side with ground truth"""
    device = get_device()
    model.eval()
    
    # Get samples
    samples = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= num_samples:
                break
            
            x = x.to(device)
            pred = model(x)
            samples.append((x.cpu().numpy(), y.cpu().numpy(), pred.cpu().numpy()))
    
    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, (x, y, pred) in enumerate(samples):
        # Input
        if x.shape[1] == 3:  # Regular RGB input
            axes[i, 0].imshow(np.transpose(x[0], (1, 2, 0)))
        else:  # Multi-channel input, show first 3 channels as RGB
            axes[i, 0].imshow(np.transpose(x[0, :3], (1, 2, 0)))
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # Ground truth
        if y.shape[1] == 3:  # RGB output
            axes[i, 1].imshow(np.transpose(y[0], (1, 2, 0)))
        else:
            axes[i, 1].imshow(y[0])
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        if pred.shape[1] == 3:  # RGB output
            axes[i, 2].imshow(np.transpose(pred[0], (1, 2, 0)))
        else:
            axes[i, 2].imshow(pred[0])
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction visualization saved to {save_path}")
    
    plt.show()

def create_time_series_animation(data, title, save_path=None, fps=2):
    """
    Create an animation from a time series of images.
    
    Args:
        data: Time series data with shape [time_steps, height, width, channels]
        title: Title of the animation
        save_path: Path to save the animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        if data[frame].shape[-1] == 3:  # RGB image
            ax.imshow(data[frame])
        else:  # Grayscale image
            ax.imshow(data[frame], cmap='gray')
        ax.set_title(f"{title} - Frame {frame}")
        ax.axis('off')
        return ax,
    
    ani = FuncAnimation(fig, update, frames=len(data), blit=True, interval=1000/fps)
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
    
    plt.close()
    return ani

def ensure_dir(dir_path):
    """Ensure a directory exists, creating it if needed"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def prepare_for_lstm_unet(X_data, y_data, time_steps=23):
    """
    Reshape data for LSTM-UNet models that require temporal information.
    
    Args:
        X_data: Input data with shape [batch, channels, height, width]
        y_data: Target data with shape [batch, channels, height, width]
        time_steps: Number of time steps to reshape the data into
        
    Returns:
        Reshaped X_data with shape [batch, time_steps, height, width, channels_per_step]
    """
    batch_size, channels, height, width = X_data.shape
    
    # Calculate channels per time step
    channels_per_step = channels // time_steps
    
    # Reshape the data
    X_reshaped = X_data.reshape(batch_size, time_steps, channels_per_step, height, width)
    X_reshaped = X_reshaped.transpose(0, 1, 3, 4, 2)  # [batch, time_steps, height, width, channels_per_step]
    
    return X_reshaped, y_data

def calculate_metrics(y_pred, y_true):
    """
    Calculate evaluation metrics between predicted and ground truth values.
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    # Mean Squared Error
    mse = np.mean(np.square(y_pred - y_true))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_pred - y_true))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }