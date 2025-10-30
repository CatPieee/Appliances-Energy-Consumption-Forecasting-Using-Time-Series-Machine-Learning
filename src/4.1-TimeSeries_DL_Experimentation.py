import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import math
from typing import Tuple, Optional

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
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

def evaluate_model(model, test_loader, scaler_y):
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

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_results(train_losses, val_losses, predictions, actuals, metrics, save_path):
    """
    Create comprehensive visualization of results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training history
    axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.7)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.7)
    axes[0, 0].set_title('Transformer Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Predictions vs Actuals (time series)
    sample_size = min(500, len(predictions))
    indices = np.random.choice(len(predictions), sample_size, replace=False)
    indices = np.sort(indices)
    
    axes[0, 1].plot(actuals[indices], label='Actual', alpha=0.7)
    axes[0, 1].plot(predictions[indices], label='Predicted', alpha=0.7)
    axes[0, 1].set_title('Transformer: Actual vs Predicted (Sample)')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Energy Consumption (Wh)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1, 0].scatter(actuals, predictions, alpha=0.5, s=1)
    axes[1, 0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[1, 0].set_title('Transformer: Predicted vs Actual')
    axes[1, 0].set_xlabel('Actual Energy Consumption (Wh)')
    axes[1, 0].set_ylabel('Predicted Energy Consumption (Wh)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = predictions - actuals
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Transformer: Residuals Distribution')
    axes[1, 1].set_xlabel('Residuals (Wh)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add metrics text
    metrics_text = f"""Transformer Model Performance:
MSE: {metrics['MSE']:.2f}
RMSE: {metrics['RMSE']:.2f}
MAE: {metrics['MAE']:.2f}
R²: {metrics['R2']:.4f}"""
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to: {save_path}")

def compare_with_previous_models(transformer_metrics):
    """
    Compare Transformer results with previous models
    """
    # Load previous results if available
    results_file = 'results/timeseries_model_results.csv'
    
    if os.path.exists(results_file):
        previous_results = pd.read_csv(results_file, index_col=0)
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        # Add Transformer results
        transformer_df = pd.DataFrame([transformer_metrics], index=['Transformer'])
        all_results = pd.concat([previous_results, transformer_df])
        
        # Sort by R² score (descending)
        all_results_sorted = all_results.sort_values('R2', ascending=False)
        
        print(all_results_sorted.round(4))
        
        # Find best model
        best_model = all_results_sorted.index[0]
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   R² Score: {all_results_sorted.loc[best_model, 'R2']:.4f}")
        print(f"   RMSE: {all_results_sorted.loc[best_model, 'RMSE']:.2f}")
        
        # Save updated results
        all_results.to_csv(results_file)
        print(f"\nUpdated results saved to: {results_file}")
        
        return all_results_sorted
    else:
        print("Previous model results not found. Only showing Transformer results.")
        transformer_df = pd.DataFrame([transformer_metrics], index=['Transformer'])
        print(transformer_df.round(4))
        return transformer_df

def main():
    """
    Main execution function
    """
    print("="*60)
    print("TIME SERIES FORECASTING WITH TRANSFORMER MODEL")
    print("="*60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data_path = 'data/energydata_complete_cleaned.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape[0]} records, {df.shape[1]} features")
    
    # Prepare features and target
    target_column = 'Appliances'
    # Exclude date column and target column from features
    exclude_columns = ['date', target_column]
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Ensure all feature columns are numeric
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    print(f"After cleaning: {df.shape[0]} records, {df.shape[1]} features")
    
    # Scale features and target separately
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Create a copy with only numeric columns
    df_scaled = df[feature_columns + [target_column]].copy()
    df_scaled[feature_columns] = scaler_X.fit_transform(df_scaled[feature_columns])
    df_scaled[target_column] = scaler_y.fit_transform(df_scaled[[target_column]])
    
    # Split data chronologically (70% train, 15% val, 15% test)
    n = len(df_scaled)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_data = df_scaled[:train_size]
    val_data = df_scaled[train_size:train_size + val_size]
    test_data = df_scaled[train_size + val_size:]
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Create datasets and dataloaders
    sequence_length = 24  # Use 24 hours to predict next hour
    batch_size = 64
    
    train_dataset = create_sequences(train_data, sequence_length, target_column)
    val_dataset = create_sequences(val_data, sequence_length, target_column)
    test_dataset = create_sequences(test_data, sequence_length, target_column)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nSequence length: {sequence_length}")
    print(f"Batch size: {batch_size}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Initialize model
    print("\n2. Initializing Transformer model...")
    model = TimeSeriesTransformer(
        input_dim=len(feature_columns),
        d_model=64,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n3. Training model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=100, learning_rate=0.001
    )
    
    # Evaluate model
    print("\n4. Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader, scaler_y)
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    
    print("\n" + "="*40)
    print("TRANSFORMER MODEL RESULTS")
    print("="*40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create results directory if it doesn't exist
    os.makedirs('results/prediction_plots', exist_ok=True)
    
    # Plot results
    print("\n5. Generating visualizations...")
    plot_results(
        train_losses, val_losses, predictions, actuals, metrics,
        'results/prediction_plots/transformer_results.png'
    )
    
    # Compare with previous models
    print("\n6. Comparing with previous models...")
    comparison_results = compare_with_previous_models(metrics)
    
    # Save detailed results
    results_summary = {
        'Model': 'Transformer',
        'Architecture': 'Multi-head Attention with Positional Encoding',
        'Sequence Length': sequence_length,
        'Parameters': trainable_params,
        'Training Epochs': len(train_losses),
        'Best Validation Loss': min(val_losses),
        **metrics
    }
    
    # Save to CSV
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv('results/transformer_detailed_results.csv', index=False)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✅ Model trained and evaluated")
    print(f"✅ Results saved to results/prediction_plots/transformer_results.png")
    print(f"✅ Detailed results saved to results/transformer_detailed_results.csv")
    print(f"✅ Model comparison updated in results/timeseries_model_results.csv")
    
    # Clean up temporary files
    if os.path.exists('best_transformer_model.pth'):
        os.remove('best_transformer_model.pth')
    
    return model, metrics, comparison_results

if __name__ == "__main__":
    main()
