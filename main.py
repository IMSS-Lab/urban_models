#!/usr/bin/env python
"""
Main script for training urban models.
This script will load data, train different models, and save the results.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Import custom modules
from src.models.unet import UNet
from src.models.lstm_unet import LSTMUNet, BidirectionalLSTMUNet
from src.models.cgan import Generator, Discriminator
from src.data.prepare import (
    load_and_preprocess_agriculture, 
    load_and_preprocess_precipitation,
    load_and_preprocess_temperature, 
    load_and_preprocess_urbanization, 
    preprocess_data,
    prepare_model_data
)
from src.data.dataset import UrbanCropDataset, TemporalUrbanCropDataset
from src.training.trainer import ModelTrainer, LSTMUNetTrainer, CGANTrainer, model_checkpoint, early_stopping
from src.utils.utils import set_seed, get_device, ensure_dir, count_parameters, visualize_prediction

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train urban models')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data files')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--visualize_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'lstm-unet', 'bilstm-unet', 'cgan'],
                        help='Type of model to train')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    
    # Other arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Save only the best model during training')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    return parser.parse_args()

def load_data(args):
    """
    Load and preprocess data.
    If preprocessed data exists, load it. Otherwise, process from raw data.
    """
    # Check if preprocessed data exists
    x_path = os.path.join(args.processed_dir, 'years_array_32_segmented_prevUrb.npy')
    y_path = os.path.join(args.processed_dir, 'crops_array_32_segmented_prevUrb.npy')
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        print("Loading preprocessed data...")
        X_data = np.load(x_path)
        y_data = np.load(y_path)
    else:
        print("Preprocessed data not found. Processing raw data...")
        ensure_dir(args.processed_dir)
        ensure_dir(args.visualize_dir)
        
        # Load the different data types
        agriculture = load_and_preprocess_agriculture(
            args.data_dir, 
            os.path.join(args.visualize_dir, 'agriculture')
        )
        precipitation = load_and_preprocess_precipitation(
            args.data_dir, 
            os.path.join(args.visualize_dir, 'precipitation')
        )
        temperature = load_and_preprocess_temperature(
            args.data_dir, 
            os.path.join(args.visualize_dir, 'temperature')
        )
        urbanization = load_and_preprocess_urbanization(
            args.data_dir, 
            os.path.join(args.visualize_dir, 'urbanization')
        )
        
        # Resize to 32x32
        precipitation, urbanization, agriculture, temperature = preprocess_data(
            precipitation, urbanization, agriculture, temperature, new_size=(32, 32)
        )
        
        # Prepare data for model training
        X_data, y_data = prepare_model_data(
            precipitation, urbanization, agriculture, temperature, args.processed_dir
        )
    
    print(f"X_data shape: {X_data.shape}")
    print(f"y_data shape: {y_data.shape}")
    
    return X_data, y_data

def create_datasets(X_data, y_data, args):
    """Create train, validation, and test datasets"""
    # Determine the number of samples for each split
    total_samples = len(X_data)
    test_size = int(total_samples * args.test_split)
    train_size = total_samples - test_size
    val_size = int(train_size * args.val_split)
    train_size = train_size - val_size
    
    # Create datasets based on model type
    if args.model_type in ['lstm-unet', 'bilstm-unet']:
        # Reshape the data for LSTM processing
        dataset = TemporalUrbanCropDataset(X_data, y_data, time_steps=23)
    else:
        # Regular dataset for U-Net and CGAN
        dataset = UrbanCropDataset(X_data, y_data)
    
    # Split into train, validation, and test sets
    train_dataset, test_dataset = random_split(
        dataset, [train_size + val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, test_loader

def create_model(args):
    """Create a model based on the specified type"""
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    
    if args.model_type == 'unet':
        # Standard U-Net model
        model = UNet(in_channels=15, out_channels=3)
        print(f"Created U-Net model with {count_parameters(model)} parameters")
        
        # Create trainer
        trainer = ModelTrainer(model, device=device)
        trainer.compile(optimizer='adam', learning_rate=args.lr, criterion='mse')
        
        return model, trainer
    
    elif args.model_type == 'lstm-unet':
        # LSTM-UNet hybrid model
        model = LSTMUNet(input_shape=(23, 32, 32, 3), lstm_units=16, unet_filters=16)
        print(f"Created LSTM-UNet model with {count_parameters(model)} parameters")
        
        # Create trainer
        trainer = LSTMUNetTrainer(model, device=device)
        trainer.compile(optimizer='adam', learning_rate=args.lr, criterion='mse')
        
        return model, trainer
    
    elif args.model_type == 'bilstm-unet':
        # Bidirectional LSTM-UNet model
        model = BidirectionalLSTMUNet(input_shape=(23, 32, 32, 3), lstm_units=16, unet_filters=16)
        print(f"Created Bidirectional LSTM-UNet model with {count_parameters(model)} parameters")
        
        # Create trainer
        trainer = LSTMUNetTrainer(model, device=device)
        trainer.compile(optimizer='adam', learning_rate=args.lr, criterion='mse')
        
        return model, trainer
    
    elif args.model_type == 'cgan':
        # Conditional GAN model
        z_size = 100
        generator_layer_size = [256, 512, 1024]
        discriminator_layer_size = [1024, 512, 256]
        class_num = 10  # Number of classes/conditions
        img_size = 32
        
        generator = Generator(z_size, class_num, generator_layer_size, img_size)
        discriminator = Discriminator(img_size, class_num, discriminator_layer_size)
        
        print(f"Created Generator with {count_parameters(generator)} parameters")
        print(f"Created Discriminator with {count_parameters(discriminator)} parameters")
        
        # Create trainer
        trainer = CGANTrainer(generator, discriminator, z_size=z_size, class_num=class_num, device=device)
        trainer.compile(optimizer='adam', learning_rate=args.lr)
        
        return (generator, discriminator), trainer
    
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

def train_model(model, trainer, train_loader, val_loader, args):
    """Train the model"""
    # Prepare callbacks
    callbacks = []
    
    # Model checkpoint callback
    if args.save_best_only:
        checkpoint_path = os.path.join(args.model_dir, f"{args.model_type}_best.pth")
        checkpoint_callback = model_checkpoint(
            checkpoint_path, 
            monitor='val_loss' if val_loader else 'train_loss',
            save_best_only=True
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if args.early_stopping:
        early_stopping_callback = early_stopping(
            patience=args.patience,
            monitor='val_loss' if val_loader else 'train_loss'
        )
        callbacks.append(early_stopping_callback)
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    
    try:
        if args.model_type == 'cgan':
            # Special case for CGAN
            history = trainer.fit(
                train_loader, 
                epochs=args.epochs, 
                sample_interval=5,
                save_dir=os.path.join(args.visualize_dir, 'cgan_samples')
            )
        else:
            # Standard training for other models
            history = trainer.fit(
                train_loader, 
                val_loader=val_loader, 
                epochs=args.epochs,
                callbacks=callbacks
            )
    except KeyboardInterrupt:
        print("Training interrupted by early stopping or user")
    
    # Save the final model
    if not args.save_best_only or args.model_type == 'cgan':
        model_path = os.path.join(args.model_dir, f"{args.model_type}_final.pth")
        trainer.save_model(model_path)
    
    # Plot training history
    history_path = os.path.join(args.visualize_dir, f"{args.model_type}_history.png")
    trainer.plot_history(save_path=history_path)
    
    return trainer

def evaluate_model(trainer, test_loader, args):
    """Evaluate the model on the test set"""
    if args.model_type == 'cgan':
        # Cannot easily evaluate CGAN in the same way
        print("Skipping evaluation for CGAN model")
        return
    
    print("Evaluating model on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_metrics['mae']:.4f}")
    
    # Visualize some predictions
    vis_path = os.path.join(args.visualize_dir, f"{args.model_type}_predictions.png")
    visualize_prediction(trainer.model, test_loader, num_samples=5, save_path=vis_path)

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    ensure_dir(args.model_dir)
    ensure_dir(args.visualize_dir)
    
    # Load data
    X_data, y_data = load_data(args)
    
    # Create datasets and loaders
    train_loader, val_loader, test_loader = create_datasets(X_data, y_data, args)
    
    # Create model and trainer
    model, trainer = create_model(args)
    
    # Train model
    trainer = train_model(model, trainer, train_loader, val_loader, args)
    
    # Evaluate model
    evaluate_model(trainer, test_loader, args)
    
    print("Done!")

if __name__ == "__main__":
    main()