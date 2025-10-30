"""
LSTM Model for Time Series Forecasting

This module implements LSTM (Long Short-Term Memory) networks for energy consumption prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.directions, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last time step
        if self.bidirectional:
            # For bidirectional, concatenate both directions
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Apply dropout and fully connected layers
        out = self.dropout(last_hidden)
        out = self.relu(self.fc(out))
        out = self.fc2(out)
        
        return out

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

def create_lstm_sequences(data, sequence_length=24, target_column='Appliances'):
    """
    Create sequences for LSTM prediction
    """
    dataset = TimeSeriesDataset(data, sequence_length, target_column)
    return dataset

def train_lstm_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the LSTM model
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print("Starting LSTM training...")
    
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
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    return train_losses, val_losses

def evaluate_lstm_model(model, test_loader, scaler_y):
    """
    Evaluate the LSTM model and return predictions
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