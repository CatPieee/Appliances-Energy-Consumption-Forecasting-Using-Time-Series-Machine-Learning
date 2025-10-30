# Appliances Energy Consumption Forecasting Using Time Series Machine Learning

## Project Overview

This project applies various time series machine learning methods to predict household appliance energy consumption, including traditional machine learning models, specialized time series models, and deep learning approaches.

## Dataset

### Data Source
- **Dataset URL**: [Kaggle - Appliances Energy Prediction Data Set](https://www.kaggle.com/datasets/sohommajumder21/appliances-energy-prediction-data-set)
- **Original Author**: Luis Candanedo, University of Mons (UMONS)

### Dataset Characteristics
- **Data Type**: Multivariate, Time-Series, Regression
- **Time Period**: January - May 2016 (4.5 months)
- **Data Size**: 19,735 records with 29 features
- **Target Variable**: Appliances (energy consumption in Wh)
- **Sampling Rate**: 10-minute intervals

### Data Collection Context
This experimental data was collected to create regression models of appliances energy use in a low energy building. Hourly weather data from Chievres Airport, Belgium was downloaded from Reliable Prognosis (rp5.ru) and interpolated to 10-minute intervals.

### Citations
**Primary Reference:**
Luis M. Candanedo, Veronique Feldheim, Dominique Deramaix, "Data driven prediction models of energy use of appliances in a low-energy house", Energy and Buildings, Volume 140, 1 April 2017, Pages 81-97, ISSN 0378-7788.

**Dataset Repository:**
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## Project Structure

```
├── data/                           # Data files
│   ├── energydata_complete_raw.csv     # Raw data
│   └── energydata_complete_cleaned.csv # Cleaned data
├── notebooks/                      # Jupyter notebooks
│   ├── 1.0-EDA-Descriptive_Stats.ipynb           # Exploratory Data Analysis
│   ├── 2.0-Feature_Engineering_Exploration.ipynb  # Feature Engineering
│   ├── 3.0-Traditional_ML_Experimentation.ipynb   # Traditional Machine Learning
│   ├── 4.0-TimeSeries_ML_Simplified.ipynb         # Time Series Machine Learning (Prophet & SARIMA)
│   └── 4.0-TimeSeries_ML_Experimentation_executed.ipynb  # Original version (deprecated)
├── src/                           # Source code
│   └── deep_learning_models/      # Deep learning model implementations
│       ├── __init__.py           # Module initialization
│       ├── rnn_model.py          # RNN model implementation
│       ├── lstm_model.py         # LSTM model implementation
│       ├── transformer_model.py  # Transformer model implementation
│       └── utils.py              # Deep learning utilities
├── 4.1-TimeSeries_DL_Experimentation.py          # Deep Learning Time Series Experimentation (deprecated)
├── results/                        # Result files
│   ├── eda_plots/                     # EDA plots
│   └── prediction_plots/              # Prediction result plots
├── neural_network.py               # PyTorch neural network implementation
├── requirements.txt                # Dependencies
└── README.md                      # Project documentation
```

## Experiments

### 1. Exploratory Data Analysis (EDA)
- Data distribution analysis
- Time series visualization
- Correlation analysis
- Seasonality and trend analysis

### 2. Feature Engineering
- Time feature extraction (hour, weekday, weekend)
- Cyclical feature encoding (sin/cos transformation)
- Lag feature creation
- Rolling statistics features

### 3. Traditional Machine Learning Models
- Linear Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Multi-layer Perceptron (MLP)

### 4. Time Series Machine Learning Models
- **Prophet**: Facebook's time series forecasting tool
- **SARIMA**: Seasonal Autoregressive Integrated Moving Average model

### 5. Deep Learning Models
- **RNN**: Recurrent Neural Network
- **LSTM**: Long Short-Term Memory neural network  
- **Transformer**: Attention-based time series forecasting model

Note: Deep learning models are implemented as separate Python modules in `src/deep_learning_models/` for better modularity and reusability.

## Model Performance Evaluation

All models are evaluated using the following metrics:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## Setup and Usage

### 1. Requirements
```bash
Python 3.8+
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments
Execute the notebooks in the following order:
1. `1.0-EDA-Descriptive_Stats.ipynb` - Data exploration
2. `2.0-Feature_Engineering_Exploration.ipynb` - Feature engineering
3. `3.0-Traditional_ML_Experimentation.ipynb` - Traditional machine learning
4. `4.0-TimeSeries_ML_Simplified.ipynb` - Time series machine learning (Prophet & SARIMA)

### 4. Run Deep Learning Time Series Experiments
```bash
# Run individual deep learning models
python src/deep_learning_models/rnn_model.py        # RNN model
python src/deep_learning_models/lstm_model.py       # LSTM model  
python src/deep_learning_models/transformer_model.py # Transformer model
```

### 5. Use Deep Learning Models as Modules
```python
from src.deep_learning_models import RNNModel, LSTMModel, TransformerModel

# Initialize and train models
rnn_model = RNNModel(input_dim=34, hidden_dim=128, num_layers=2)
lstm_model = LSTMModel(input_dim=34, hidden_dim=128, num_layers=2)
transformer_model = TransformerModel(input_dim=34, d_model=128, num_heads=8, num_layers=6)
```

## Key Findings

1. **Seasonal Patterns**: Appliance energy consumption shows clear daily and weekly seasonality
2. **Feature Importance**: Lag features and rolling statistics significantly improve prediction performance
3. **Model Performance**: Time series specialized models perform better at capturing temporal dependencies
4. **Deep Learning**: RNN, LSTM, and Transformer models can learn complex nonlinear temporal patterns

## Architecture Overview

This project is organized into three main categories:

1. **Traditional Machine Learning** (`notebooks/3.0-Traditional_ML_Experimentation.ipynb`)
   - Linear Regression, SVM, Decision Tree, Random Forest, XGBoost, MLP
   - All models in a single Jupyter notebook for quick experimentation

2. **Time Series Machine Learning** (`notebooks/4.0-TimeSeries_ML_Simplified.ipynb`)
   - Prophet and SARIMA models only
   - Focused on traditional time series forecasting methods

3. **Deep Learning Models** (`src/deep_learning_models/`)
   - RNN, LSTM, and Transformer models as separate Python modules
   - Modular design for better reusability and longer training times

## Results

- `results/eda_plots/`: EDA analysis plots
- `results/prediction_plots/`: Model prediction result plots
- `results/timeseries_model_results.csv`: Time series model performance comparison
- `results/transformer_detailed_results.csv`: Transformer model detailed results

## Technology Stack

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Traditional ML**: scikit-learn, xgboost
- **Time Series**: prophet, statsmodels
- **Deep Learning**: tensorflow, keras, pytorch
- **Development**: Jupyter Notebook