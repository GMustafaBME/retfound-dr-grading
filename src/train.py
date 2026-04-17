"""
Training Script for RETFound DR Grading

Handles model training with learning rate scheduling,
early stopping, and experiment logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
    
    Returns:
        avg_loss: Average training loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model on validation set.
    
    Args:
        model: PyTorch model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: torch.device
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RETFound DR model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    # TODO: Load config, initialize model, run training loop
    print(f"Training with config: {args.config}")
