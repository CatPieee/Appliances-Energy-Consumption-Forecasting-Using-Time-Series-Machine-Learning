"""
Utility functions for deep learning models

This module contains common utility functions used across different deep learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

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

def plot_training_history(train_losses, val_losses, model_name, save_path=None):
    """
    Plot training and validation loss history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_prediction_results(predictions, actuals, model_name, save_path=None):
    """
    Create comprehensive visualization of prediction results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Predictions vs Actuals (time series sample)
    sample_size = min(500, len(predictions))
    indices = np.random.choice(len(predictions), sample_size, replace=False)
    indices = np.sort(indices)
    
    axes[0, 0].plot(actuals[indices], label='Actual', alpha=0.7)
    axes[0, 0].plot(predictions[indices], label='Predicted', alpha=0.7)
    axes[0, 0].set_title(f'{model_name}: Actual vs Predicted (Sample)')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Energy Consumption (Wh)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[0, 1].scatter(actuals, predictions, alpha=0.5, s=1)
    axes[0, 1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 1].set_title(f'{model_name}: Predicted vs Actual')
    axes[0, 1].set_xlabel('Actual Energy Consumption (Wh)')
    axes[0, 1].set_ylabel('Predicted Energy Consumption (Wh)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals distribution
    residuals = predictions - actuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title(f'{model_name}: Residuals Distribution')
    axes[1, 0].set_xlabel('Residuals (Wh)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals over time
    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.5, s=1)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].set_title(f'{model_name}: Residuals Over Time')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Residuals (Wh)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_model_results(model_name, metrics, sequence_length, parameters, training_epochs, 
                      best_val_loss, architecture, results_dir='results'):
    """
    Save detailed model results to CSV file
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    results_summary = {
        'Model': model_name,
        'Architecture': architecture,
        'Sequence Length': sequence_length,
        'Parameters': parameters,
        'Training Epochs': training_epochs,
        'Best Validation Loss': best_val_loss,
        **metrics
    }
    
    # Save to CSV
    results_file = os.path.join(results_dir, f'{model_name.lower()}_results.csv')
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv(results_file, index=False)
    
    print(f"✅ Results saved to: {results_file}")
    return results_file

def compare_models(results_dir='results'):
    """
    Compare all model results in the results directory
    """
    import glob
    
    # Find all model result files
    result_files = glob.glob(os.path.join(results_dir, '*_results.csv'))
    
    if not result_files:
        print("No model result files found!")
        return None
    
    # Combine all results
    all_results = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Sort by R² score (descending)
        if 'R2' in combined_results.columns:
            combined_results = combined_results.sort_values('R2', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        print(combined_results.round(4))
        
        # Find best model
        if 'R2' in combined_results.columns and not combined_results.empty:
            best_model = combined_results.iloc[0]
            print(f"\n🏆 Best Model: {best_model['Model']}")
            print(f"   R² Score: {best_model['R2']:.4f}")
            print(f"   RMSE: {best_model['RMSE']:.2f}")
        
        return combined_results
    
    return None