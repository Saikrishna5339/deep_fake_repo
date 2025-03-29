import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.genconvit import GenConViT, Discriminator
from utils.data_loader import create_data_loaders, get_transforms

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    
    # Enable gradient checkpointing to save memory
    model.enable_checkpointing()
    
    # Set CUDA memory allocator settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate training metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_prec = precision_score(train_labels, train_preds)
        train_rec = recall_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        model.disable_checkpointing()  # Disable checkpointing during validation
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Re-enable checkpointing for next epoch
        model.enable_checkpointing()
        
        # Calculate validation metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds)
        val_rec = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        # Print metrics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Train Acc: {train_acc:.4f}, Train Prec: {train_prec:.4f}, Train Rec: {train_rec:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Acc: {val_acc:.4f}, Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}, Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with validation accuracy: {best_val_acc:.4f}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set CUDA memory allocator settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    # Create model
    model = GenConViT(num_classes=2).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loaders with smaller batch size
    train_loader, val_loader = create_data_loaders(
        train_dir='data/train',
        val_dir='data/val',
        batch_size=16  # Reduced from 32 to 16
    )
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        device=device
    )

if __name__ == '__main__':
    main() 