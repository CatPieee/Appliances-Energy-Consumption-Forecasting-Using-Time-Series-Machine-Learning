# Appliances Energy Consumption Forecasting Using Time Series Machine Learning

## 项目概述

本项目使用多种时间序列机器学习方法预测家电能耗，包括传统机器学习模型、时间序列专用模型和深度学习方法。

## 数据集

- **数据来源**: 家庭能耗监测数据
- **时间范围**: 2016年1月-5月
- **数据量**: 19,591条记录，37个特征
- **目标变量**: Appliances (家电能耗，单位：Wh)
- **特征类型**: 
  - 温度传感器数据 (T1-T9)
  - 湿度传感器数据 (RH_1-RH_9)
  - 天气数据 (温度、湿度、气压、风速等)
  - 时间特征 (小时、星期、周末标识等)
  - 滞后特征和滚动统计特征

## 项目结构

```
├── data/                           # 数据文件
│   ├── energydata_complete_raw.csv     # 原始数据
│   └── energydata_complete_cleaned.csv # 清洗后数据
├── notebooks/                      # Jupyter notebooks
│   ├── 1.0-EDA-Descriptive_Stats.ipynb           # 探索性数据分析
│   ├── 2.0-Feature_Engineering_Exploration.ipynb  # 特征工程
│   ├── 3.0-Traditional_ML_Experimentation.ipynb   # 传统机器学习
│   └── 4.0-TimeSeries_ML_Experimentation.ipynb    # 时间序列机器学习
├── results/                        # 结果文件
│   ├── eda_plots/                     # EDA图表
│   └── prediction_plots/              # 预测结果图表
├── neural_network.py               # PyTorch神经网络实现
├── requirements.txt                # 依赖包列表
└── README.md                      # 项目说明
```

## 实验内容

### 1. 探索性数据分析 (EDA)
- 数据分布分析
- 时间序列可视化
- 相关性分析
- 季节性和趋势分析

### 2. 特征工程
- 时间特征提取 (小时、星期、周末)
- 周期性特征编码 (sin/cos变换)
- 滞后特征创建
- 滚动统计特征

### 3. 传统机器学习模型
- 线性回归 (Linear Regression)
- 随机森林 (Random Forest)
- XGBoost
- 支持向量机 (SVM)
- 神经网络 (MLP)

### 4. 时间序列机器学习模型
- **Prophet**: Facebook开发的时间序列预测工具
- **ARIMA**: 自回归积分滑动平均模型
- **LSTM**: 长短期记忆神经网络

### 5. 深度学习模型
- PyTorch实现的多层神经网络
- TensorFlow/Keras实现的LSTM网络

## 模型性能对比

所有模型使用以下评估指标：
- **MSE** (均方误差)
- **RMSE** (均方根误差)
- **MAE** (平均绝对误差)
- **R²** (决定系数)

## 安装和运行

### 1. 环境要求
```bash
Python 3.8+
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行实验
按顺序运行以下notebook：
1. `1.0-EDA-Descriptive_Stats.ipynb` - 数据探索
2. `2.0-Feature_Engineering_Exploration.ipynb` - 特征工程
3. `3.0-Traditional_ML_Experimentation.ipynb` - 传统机器学习
4. `4.0-TimeSeries_ML_Experimentation.ipynb` - 时间序列机器学习

### 4. 运行神经网络模型
```bash
python neural_network.py
```

## 主要发现

1. **季节性模式**: 家电能耗具有明显的日内和周内季节性
2. **特征重要性**: 滞后特征和滚动统计特征对预测性能提升显著
3. **模型性能**: 时间序列专用模型在捕获时间依赖性方面表现更好
4. **深度学习**: LSTM能够学习复杂的非线性时间模式

## 结果文件

- `results/eda_plots/`: EDA分析图表
- `results/prediction_plots/`: 模型预测结果图表
- `results/timeseries_model_results.csv`: 时间序列模型性能对比

## 技术栈

- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **传统机器学习**: scikit-learn, xgboost
- **时间序列**: prophet, statsmodels
- **深度学习**: tensorflow, keras, pytorch
- **开发环境**: Jupyter Notebook

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License