#!/usr/bin/env python
"""
Analysis script for urban models.
This script loads trained models and analyzes their performance.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Import custom modules
from models.unet import UNet
from models.lstm_unet import LSTMUNet, BidirectionalLSTMUNet
from models.cgan import Generator, Discriminator
from data.prepare import prepare_model_data
from data.dataset import UrbanCropDataset, TemporalUrbanCropDataset
from training import ModelTrainer, LSTMUNetTrainer, CGANTrainer
from utils import (
    set_seed, get_device, ensure_dir, visualize_prediction, 
    calculate_metrics, create_time_series_animation
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze urban models')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing the processed data files')
    parser.add_argument('--output_dir', type=str, default='analysis',
                        help='Directory to save analysis results')
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--model_type', type=str, default='all',
                        choices=['unet', 'lstm-unet', 'bilstm-unet', 'cgan', 'all'],
                        help='Type of model to analyze')
    
    # Analysis arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare performance of different models')
    
    # Other arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()

def load_data(args):
    """Load the processed data"""
    print("Loading data...")
    
    x_path = os.path.join(args.data_dir, 'years_array_32_segmented_prevUrb.npy')
    y_path = os.path.join(args.data_dir, 'crops_array_32_segmented_prevUrb.npy')
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Data files not found in {args.data_dir}")
    
    X_data = np.load(x_path)
    y_data = np.load(y_path)
    
    print(f"X_data shape: {X_data.shape}")
    print(f"y_data shape: {y_data.shape}")
    
    return X_data, y_data

def create_test_loader(X_data, y_data, args, model_type):
    """Create a test data loader"""
    # Split into train and test sets
    total_samples = len(X_data)
    test_size = int(total_samples * 0.2)  # 20% for testing
    
    # Create dataset based on model type
    if model_type in ['lstm-unet', 'bilstm-unet']:
        # Reshape the data for LSTM processing
        dataset = TemporalUrbanCropDataset(X_data, y_data, time_steps=23)
    else:
        # Regular dataset for U-Net and CGAN
        dataset = UrbanCropDataset(X_data, y_data)
    
    # Create train/test split
    _, test_dataset = random_split(
        dataset, [total_samples - test_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    return test_loader

def load_model(args, model_type):
    """Load a trained model"""
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    
    # Check for model file
    model_path = os.path.join(args.model_dir, f"{model_type}_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, f"{model_type}_final.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create and load model
    if model_type == 'unet':
        model = UNet(in_channels=15, out_channels=3)
        trainer = ModelTrainer(model, device=device)
        
    elif model_type == 'lstm-unet':
        model = LSTMUNet(input_shape=(23, 32, 32, 3), lstm_units=16, unet_filters=16)
        trainer = LSTMUNetTrainer(model, device=device)
        
    elif model_type == 'bilstm-unet':
        model = BidirectionalLSTMUNet(input_shape=(23, 32, 32, 3), lstm_units=16, unet_filters=16)
        trainer = LSTMUNetTrainer(model, device=device)
        
    elif model_type == 'cgan':
        z_size = 100
        generator_layer_size = [256, 512, 1024]
        discriminator_layer_size = [1024, 512, 256]
        class_num = 10
        img_size = 32
        
        generator = Generator(z_size, class_num, generator_layer_size, img_size)
        discriminator = Discriminator(img_size, class_num, discriminator_layer_size)
        
        trainer = CGANTrainer(generator, discriminator, z_size=z_size, class_num=class_num, device=device)
        trainer.compile(optimizer='adam', learning_rate=0.0001)
        trainer.load_model(os.path.join(args.model_dir, model_type))
        
        return (generator, discriminator), trainer
    
    # Compile trainer (default parameters will be overridden by loaded state)
    trainer.compile(optimizer='adam', learning_rate=0.001, criterion='mse')
    
    # Load model state
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded {model_type} model from {model_path}")
    
    return model, trainer

def analyze_model_predictions(model, test_loader, args, model_type):
    """Analyze model predictions"""
    device = get_device()
    model.to(device)
    model.eval()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, model_type)
    ensure_dir(output_dir)
    
    # Get samples for analysis
    all_inputs = []
    all_targets = []
    all_preds = []
    
    print(f"Analyzing {model_type} model predictions...")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if batch_idx >= args.num_samples:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            # Make prediction
            pred = model(x)
            
            # Store for analysis
            all_inputs.append(x.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
    
    # Combine batches
    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    
    # Visualize predictions
    visualize_path = os.path.join(output_dir, "predictions.png")
    fig, axes = plt.subplots(min(args.num_samples, len(inputs)), 3, figsize=(15, 5*min(args.num_samples, len(inputs))))
    
    for i in range(min(args.num_samples, len(inputs))):
        # Input (use first three channels)
        if inputs.shape[1] == 3:  # Regular RGB input
            axes[i, 0].imshow(np.transpose(inputs[i], (1, 2, 0)))
        else:  # Multi-channel input, show first 3 channels as RGB
            if len(inputs.shape) == 5:  # LSTM data has shape [batch, time, height, width, channels]
                axes[i, 0].imshow(inputs[i, 0, :, :, :])
            else:
                axes[i, 0].imshow(np.transpose(inputs[i, :3], (1, 2, 0)))
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # Ground truth
        if targets.shape[1] == 3:  # RGB output
            axes[i, 1].imshow(np.transpose(targets[i], (1, 2, 0)))
        else:
            axes[i, 1].imshow(targets[i])
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        if preds.shape[1] == 3:  # RGB output
            axes[i, 2].imshow(np.transpose(preds[i], (1, 2, 0)))
        else:
            axes[i, 2].imshow(preds[i])
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(visualize_path)
    plt.close()
    
    print(f"Saved prediction visualization to {visualize_path}")
    
    # Calculate and visualize error distribution
    if targets.shape == preds.shape:
        error = np.abs(targets - preds)
        
        # Create error heatmap (average across samples)
        error_mean = np.mean(error, axis=0)
        if len(error_mean.shape) == 3:  # Multi-channel output
            error_mean = np.mean(error_mean, axis=0)  # Average across channels
        
        plt.figure(figsize=(10, 8))
        plt.imshow(error_mean, cmap='hot')
        plt.colorbar(label='Mean Absolute Error')
        plt.title(f'{model_type} - Error Distribution')
        plt.savefig(os.path.join(output_dir, "error_heatmap.png"))
        plt.close()
        
        # Error histogram
        error_flat = error.flatten()
        plt.figure(figsize=(10, 6))
        sns.histplot(error_flat, bins=50)
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title(f'{model_type} - Error Histogram')
        plt.savefig(os.path.join(output_dir, "error_histogram.png"))
        plt.close()
        
        # Calculate metrics
        metrics = calculate_metrics(preds, targets)
        print(f"Model: {model_type}, MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        # Save metrics to file
        with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
    
    return metrics if targets.shape == preds.shape else None

def analyze_feature_space(model, test_loader, args, model_type):
    """
    Analyze the feature space learned by the model.
    Extract features from the penultimate layer and visualize using PCA and t-SNE.
    """
    # Only for non-CGAN models
    if model_type == 'cgan':
        return
    
    device = get_device()
    model.to(device)
    model.eval()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, model_type)
    ensure_dir(output_dir)
    
    # Function to extract features from penultimate layer
    def extract_features(model, x):
        # This function needs to be customized for each model type
        # Here we'll just use the final output as a placeholder
        return model(x).cpu().numpy().reshape(x.size(0), -1)
    
    # Get features for all test samples
    all_features = []
    all_labels = []  # We'll use target values as "labels" for visualization
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if batch_idx >= 10:  # Limit the number of batches for speed
                break
            
            x = x.to(device)
            features = extract_features(model, x)
            
            # Use mean of target as label
            y_mean = y.view(y.size(0), -1).mean(dim=1).cpu().numpy()
            
            all_features.append(features)
            all_labels.append(y_mean)
    
    # Combine batches
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Use PCA to reduce dimensionality
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Plot PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Target Mean Value')
    plt.title(f'{model_type} - PCA of Features')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(output_dir, "pca_features.png"))
    plt.close()
    
    # Use t-SNE to visualize feature space
    tsne = TSNE(n_components=2, random_state=args.seed)
    features_tsne = tsne.fit_transform(features)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Target Mean Value')
    plt.title(f'{model_type} - t-SNE of Features')
    plt.savefig(os.path.join(output_dir, "tsne_features.png"))
    plt.close()
    
    print(f"Saved feature space visualizations for {model_type}")

def compare_models(args, metrics_dict):
    """Compare the performance of different models"""
    if len(metrics_dict) <= 1:
        print("Not enough models to compare")
        return
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Create a DataFrame from metrics
    data = []
    for model_type, metrics in metrics_dict.items():
        data.append({
            'model': model_type,
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse']
        })
    
    df = pl.DataFrame(data)
    
    # Save metrics to CSV
    df.write_csv(os.path.join(args.output_dir, "model_comparison.csv"))
    
    # Create bar plot for each metric
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['mse', 'mae', 'rmse']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i+1)
        
        # Convert to pandas for easier plotting with seaborn
        pd_df = df.to_pandas()
        sns.barplot(x='model', y=metric, data=pd_df)
        plt.title(f'Comparison of {metric.upper()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(os.path.join(args.output_dir, "model_comparison.png"))
    plt.close()
    
    # Print comparison table
    print("\nModel Comparison:")
    print(df)
    
    # Find best model for each metric
    best_models = {}
    for metric in metrics_to_plot:
        if metric in ['mse', 'mae', 'rmse']:
            best_idx = df[metric].argmin()
            best_model = df['model'][best_idx]
            best_models[metric] = best_model
    
    print("\nBest Models:")
    for metric, model in best_models.items():
        print(f"  {metric.upper()}: {model}")

def analyze_cgan(generator, discriminator, args):
    """Analyze the CGAN model"""
    device = get_device()
    generator.to(device)
    generator.eval()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, "cgan")
    ensure_dir(output_dir)
    
    # Generate samples for each class
    z_size = 100
    class_num = 10  # Adjust based on your model
    
    with torch.no_grad():
        # Generate a grid of samples
        z = torch.randn(class_num * 10, z_size).to(device)
        labels = torch.LongTensor(np.repeat(np.arange(class_num), 10)).to(device)
        
        # Generate images
        samples = generator(z, labels)
        samples = samples.cpu().detach()
        
        # Create grid
        fig, axs = plt.subplots(class_num, 10, figsize=(20, 20))
        for i in range(class_num):
            for j in range(10):
                idx = i * 10 + j
                img = samples[idx].squeeze().numpy()
                if axs.ndim == 1:
                    ax = axs[j]
                else:
                    ax = axs[i, j]
                ax.imshow(img, cmap='gray')
                if j == 0:
                    ax.set_ylabel(f'Class {i}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "generated_samples.png"))
        plt.close()
    
    print("Saved CGAN analysis to", output_dir)

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Load data
    X_data, y_data = load_data(args)
    
    # Determine which models to analyze
    if args.model_type == 'all':
        model_types = ['unet', 'lstm-unet', 'bilstm-unet', 'cgan']
    else:
        model_types = [args.model_type]
    
    # Track metrics for comparison
    metrics_dict = {}
    
    # Analyze each model
    for model_type in model_types:
        try:
            # Create test loader
            test_loader = create_test_loader(X_data, y_data, args, model_type)
            
            # Load model
            model, trainer = load_model(args, model_type)
            
            # Special case for CGAN
            if model_type == 'cgan':
                generator, discriminator = model
                analyze_cgan(generator, discriminator, args)
                continue
            
            # Analyze predictions
            metrics = analyze_model_predictions(model, test_loader, args, model_type)
            if metrics:
                metrics_dict[model_type] = metrics
            
            # Analyze feature space
            analyze_feature_space(model, test_loader, args, model_type)
            
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Skipping {model_type}: {e}")
    
    # Compare models if requested
    if args.compare_models and len(metrics_dict) > 1:
        compare_models(args, metrics_dict)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()