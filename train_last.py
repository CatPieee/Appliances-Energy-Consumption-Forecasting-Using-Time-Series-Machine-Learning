import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("Step 1: Loading and preprocessing data...")

energy_df = pd.read_csv('./data/energydata_complete_cleaned.csv',
                        parse_dates=['date'],
                        index_col='date')

energy_df.sort_index(inplace=True)
print(f'Dataset shape: {energy_df.shape}')

feature_columns = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
                   'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9',
                   'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
                   'Visibility', 'Tdewpoint', 'hour_of_day', 'day_of_week',
                   'is_weekend', 'hour_sin', 'hour_cos', 'day_of_week_sin',
                   'day_of_week_cos', 'Appliances_lag1', 'Appliances_rolling_mean_6']

target_column = 'Appliances'

print(f'Number of features: {len(feature_columns)}')

X_data = energy_df[feature_columns]
y_data = energy_df[target_column]

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
    X_data, y_data,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

print(f'Training set shape: {X_train_data.shape}')
print(f'Test set shape: {X_test_data.shape}')

data_scaler = StandardScaler()
X_train_scaled = data_scaler.fit_transform(X_train_data)
X_test_scaled = data_scaler.transform(X_test_data)

print("Data preparation completed!\n")

print("=" * 50)
print("Model 1: Linear Regression")
print("=" * 50)

linear_model = LinearRegression()

print("Training Linear Regression model...")
linear_model.fit(X_train_data, y_train_data)

linear_predictions = linear_model.predict(X_test_data)

linear_mse = mean_squared_error(y_test_data, linear_predictions)
linear_mae = mean_absolute_error(y_test_data, linear_predictions)
linear_r2 = r2_score(y_test_data, linear_predictions)

print("Linear Regression training completed!")
print(f"MSE: {linear_mse:.4f}")
print(f"MAE: {linear_mae:.4f}")
print(f"R²: {linear_r2:.4f}")

model_results = {
    'Linear Regression': {
        'predictions': linear_predictions,
        'MSE': linear_mse,
        'MAE': linear_mae,
        'R2': linear_r2
    }
}

print("\n")

print("=" * 50)
print("Model 2: Random Forest Regression")
print("=" * 50)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")
rf_model.fit(X_train_data, y_train_data)

rf_predictions = rf_model.predict(X_test_data)

rf_mse = mean_squared_error(y_test_data, rf_predictions)
rf_mae = mean_absolute_error(y_test_data, rf_predictions)
rf_r2 = r2_score(y_test_data, rf_predictions)

print("Random Forest training completed!")
print(f"MSE: {rf_mse:.4f}")
print(f"MAE: {rf_mae:.4f}")
print(f"R²: {rf_r2:.4f}")

model_results['Random Forest'] = {
    'predictions': rf_predictions,
    'MSE': rf_mse,
    'MAE': rf_mae,
    'R2': rf_r2
}

print("\n")

print("=" * 50)
print("Model 3: Support Vector Regression")
print("=" * 50)

svr_model = SVR(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)

print("Training Support Vector Regression model...")
svr_model.fit(X_train_scaled, y_train_data)

svr_predictions = svr_model.predict(X_test_scaled)

svr_mse = mean_squared_error(y_test_data, svr_predictions)
svr_mae = mean_absolute_error(y_test_data, svr_predictions)
svr_r2 = r2_score(y_test_data, svr_predictions)

print("Support Vector Regression training completed!")
print(f"MSE: {svr_mse:.4f}")
print(f"MAE: {svr_mae:.4f}")
print(f"R²: {svr_r2:.4f}")

model_results['Support Vector Regression'] = {
    'predictions': svr_predictions,
    'MSE': svr_mse,
    'MAE': svr_mae,
    'R2': svr_r2
}

print("\n")

print("=" * 50)
print("Model 4: XGBoost Regression")
print("=" * 50)

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    random_state=42,
    learning_rate=0.1,
    max_depth=6
)

print("Training XGBoost model...")
xgb_model.fit(X_train_data, y_train_data)

xgb_predictions = xgb_model.predict(X_test_data)

xgb_mse = mean_squared_error(y_test_data, xgb_predictions)
xgb_mae = mean_absolute_error(y_test_data, xgb_predictions)
xgb_r2 = r2_score(y_test_data, xgb_predictions)

print("XGBoost training completed!")
print(f"MSE: {xgb_mse:.4f}")
print(f"MAE: {xgb_mae:.4f}")
print(f"R²: {xgb_r2:.4f}")

model_results['XGBoost'] = {
    'predictions': xgb_predictions,
    'MSE': xgb_mse,
    'MAE': xgb_mae,
    'R2': xgb_r2
}

print("\n")

print("=" * 50)
print("Model 5: Multi-layer Perceptron Regression")
print("=" * 50)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    random_state=42,
    max_iter=500,
    early_stopping=True,
    learning_rate_init=0.001
)

print("Training Multi-layer Perceptron model...")
mlp_model.fit(X_train_scaled, y_train_data)

mlp_predictions = mlp_model.predict(X_test_scaled)

mlp_mse = mean_squared_error(y_test_data, mlp_predictions)
mlp_mae = mean_absolute_error(y_test_data, mlp_predictions)
mlp_r2 = r2_score(y_test_data, mlp_predictions)

print("Multi-layer Perceptron training completed!")
print(f"MSE: {mlp_mse:.4f}")
print(f"MAE: {mlp_mae:.4f}")
print(f"R²: {mlp_r2:.4f}")

model_results['Multi-layer Perceptron'] = {
    'predictions': mlp_predictions,
    'MSE': mlp_mse,
    'MAE': mlp_mae,
    'R2': mlp_r2
}

print("\n")

print("=" * 50)
print("Model Performance Comparison Analysis")
print("=" * 50)

results_comparison = pd.DataFrame({
    model: {
        'MSE': metrics['MSE'],
        'MAE': metrics['MAE'],
        'R2': metrics['R2']
    }
    for model, metrics in model_results.items()
}).T.round(4)

print("Model Performance Metrics Comparison:")
print(results_comparison)

print("\nGenerating visualization charts...")

plt.figure(figsize=(16, 12))
sample_count = min(100, len(y_test_data))

for idx, (model_name, results) in enumerate(model_results.items(), 1):
    plt.subplot(3, 2, idx)

    y_pred = results['predictions']

    plt.plot(range(sample_count), y_test_data.values[:sample_count],
             'steelblue', linewidth=2.5, label='True Values', alpha=0.9)
    plt.plot(range(sample_count), y_pred[:sample_count],
             'crimson', linestyle='--', linewidth=1.8, label='Predicted Values', alpha=0.8)

    plt.title(f'{model_name}', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Sample Index', fontsize=11)
    plt.ylabel('Energy Consumption', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Energy Consumption Prediction Comparison Across Models', fontsize=16, fontweight='bold', y=1.02)
plt.show()

metric_names = ['MSE', 'MAE', 'R2']
model_names = list(model_results.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, metric in enumerate(metric_names):
    metric_values = [model_results[model][metric] for model in model_names]

    bar_colors = ['lightcoral' if metric in ['MSE', 'MAE'] else 'lightgreen' for _ in model_names]

    bars = axes[i].bar(model_names, metric_values, color=bar_colors, alpha=0.8, edgecolor='grey')
    axes[i].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[i].set_ylabel(metric, fontsize=12)
    axes[i].tick_params(axis='x', rotation=35)
    axes[i].grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width() / 2., height + max(metric_values) * 0.01,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.suptitle('Machine Learning Models Performance Metrics Comparison', fontsize=16, fontweight='bold', y=1.05)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (model_name, results) in enumerate(model_results.items()):
    y_pred = results['predictions']

    axes[idx].scatter(y_test_data, y_pred, alpha=0.6, s=25, color='dodgerblue')

    min_val = min(y_test_data.min(), y_pred.min())
    max_val = max(y_test_data.max(), y_pred.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--',
                   linewidth=2, label='Perfect Prediction')

    axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('True Values', fontsize=10)
    axes[idx].set_ylabel('Predicted Values', fontsize=10)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

for idx in range(len(model_results), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.suptitle('Predicted vs True Values Scatter Plot', fontsize=16, fontweight='bold', y=1.02)
plt.show()

print("\n" + "=" * 60)
print("Model Performance Ranking")
print("=" * 60)

r2_rankings = sorted(model_results.items(), key=lambda x: x[1]['R2'], reverse=True)
print("\nR² Score Ranking (Higher is Better):")
for rank, (model, metrics) in enumerate(r2_rankings, 1):
    print(f"  Rank {rank}: {model:25} R² = {metrics['R2']:.4f}")

mse_rankings = sorted(model_results.items(), key=lambda x: x[1]['MSE'])
print("\nMSE Ranking (Lower is Better):")
for rank, (model, metrics) in enumerate(mse_rankings, 1):
    print(f"  Rank {rank}: {model:25} MSE = {metrics['MSE']:.4f}")

mae_rankings = sorted(model_results.items(), key=lambda x: x[1]['MAE'])
print("\nMAE Ranking (Lower is Better):")
for rank, (model, metrics) in enumerate(mae_rankings, 1):
    print(f"  Rank {rank}: {model:25} MAE = {metrics['MAE']:.4f}")

best_r2_model = r2_rankings[0][0]
best_mse_model = mse_rankings[0][0]
best_mae_model = mae_rankings[0][0]

print("\n" + "Best Model Summary " + "=" * 40)
print(f"  Best R² Model:  {best_r2_model} (R² = {model_results[best_r2_model]['R2']:.4f})")
print(f"  Best MSE Model: {best_mse_model} (MSE = {model_results[best_mse_model]['MSE']:.4f})")
print(f"  Best MAE Model: {best_mae_model} (MAE = {model_results[best_mae_model]['MAE']:.4f})")

top_models = {best_r2_model, best_mse_model, best_mae_model}
if len(top_models) == 1:
    overall_best = list(top_models)[0]
    print(f"\nOverall Best Model: {overall_best}")
    print(f"  This model performs best on all three metrics!")
else:
    print(f"\nDifferent models excel in different metrics:")
    print(f"  - For prediction accuracy → Choose {best_r2_model} (Highest R²)")
    print(f"  - For error control → Choose {best_mse_model} (Lowest MSE)")

print("\n" + "=" * 60)
print("Model Training and Evaluation Completed!")
print("=" * 60)