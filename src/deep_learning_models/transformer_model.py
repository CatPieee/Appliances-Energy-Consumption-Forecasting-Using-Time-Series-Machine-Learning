"""
Transformer Model for Time Series Forecasting

This module implements a Transformer-based model for energy consumption prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer model
    Adds positional information to input embeddings
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series forecasting
    """
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
    def forward(self, src, src_mask=None):
        # Input projection
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoding
        output = self.transformer_encoder(src, src_mask)
        
        # Take the last time step for prediction
        output = output[:, -1, :]  # (batch_size, d_model)
        
        # Output projection
        output = self.output_projection(output)
        
        return output

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time series data
    """
    def __init__(self, data, sequence_length, target_column='Appliances'):
        self.data = data
        self.sequence_length = sequence_length
        self.target_column = target_column
        
        # Prepare features (exclude target column)
        feature_columns = [col for col in data.columns if col != target_column]
        self.features = data[feature_columns].values.astype(np.float32)
        self.targets = data[target_column].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.sequence_length]
        # Get target (next value)
        y = self.targets[idx + self.sequence_length]
        
        # Ensure data is float32
        x = np.array(x, dtype=np.float32)
        y = np.array([y], dtype=np.float32)
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

def create_sequences(data, sequence_length=24, target_column='Appliances'):
    """
    Create sequences for time series prediction
    """
    dataset = TimeSeriesDataset(data, sequence_length, target_column)
    return dataset

def train_transformer_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the Transformer model
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    
    return train_losses, val_losses

def evaluate_transformer_model(model, test_loader, scaler_y):
    """
    Evaluate the model and return predictions
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Inverse transform predictions and actuals
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Ensure non-negative predictions
    predictions = np.maximum(predictions, 0)
    
    return predictions, actuals