import argparse
import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.predictive_network_planning.model import PredictiveNetwork
from src.utils.data_loader import PNPDataset

def train(model: nn.Module,
          train_data: PNPDataset,
          valid_data: PNPDataset,
          epochs: int,
          batch_size: int,
          lr: float,
          weight_decay: float,
          device: str,
          save_path: str) -> Tuple[List[float], List[float]]:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_loss = []
    valid_loss = []
    best_valid_loss = np.inf
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_data)
        train_loss.append(epoch_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(valid_data)
            valid_loss.append(epoch_loss)

            if epoch_loss < best_valid_loss:
                best_valid_loss = epoch_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Valid Loss: {valid_loss[-1]:.4f}")

    return train_loss, valid_loss

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_file', type=str, required=True)
    parser.add_argument('--valid_data_file', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='models/predictive_network_planning/best_model.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Load data
    train_data = PNPDataset(data_file=args.train_data_file)
    valid_data = PNPDataset(data_file=args.valid_data_file)

    # Initialize model
    model = PredictiveNetwork(input_size=train_data.input_dim,
                              hidden_size=64,
                              output_size=train_data.output_dim)
    device = torch.device(args.device)
    model.to(device)

    # Train model
    train(model=model,
          train_data=train_data,
          valid_data=valid_data,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          weight_decay=args.weight_decay,
          device=args.device,
          save_path=args.save_path)

if __name__ == "__main__":
    main()