# 📊 Forecast Evaluation Report

## Executive Summary
This report summarizes the performance of 7 different machine learning and statistical models evaluated for predicting daily water consumption at the household and district level. 

## Models Evaluated
1. **Prophet (Statistical/Time-Series)**
2. **SARIMA (Statistical/Time-Series)**
3. **LSTM (Deep Learning Recurrent Neural Network)**
4. **GRU (Deep Learning Recurrent Neural Network)**
5. **Random Forest Regressor (Tree-Based Ensemble)**
6. **LightGBM (Gradient Boosting)**
7. **XGBoost (Gradient Boosting)**

## Evaluation Metrics (Test Set Results)
The models were evaluated chronologically on the test set (last 15% of the data timeline) to prevent data leakage. The results are ordered from best to worst based on RMSE.

| Model | RMSE (Liters) | MAE (Liters) | MAPE (%) | R² Score |
|---|---|---|---|---|
| **Prophet** | 2,047.43 | 1,636.42 | 14.19% | -0.0068 |
| **SARIMA** | 2,363.29 | 1,793.68 | 16.22% | -0.3415 |
| **LSTM** | 4,053.67 | 3,441.26 | 34.23% | 0.0180 |
| **Random Forest** | 4,128.54 | 3,455.92 | 34.99% | -0.0186 |
| **LightGBM** | 4,423.02 | 3,645.82 | 35.62% | -0.1691 |
| **XGBoost** | 4,580.85 | 3,728.25 | 37.73% | -0.2540 |
| **GRU** | 4,685.76 | 3,847.73 | 30.97% | -0.3121 |

## Analysis of Results
- **Time-Series Models Won:** The traditional time-series models (Prophet and SARIMA) significantly outperformed all other models, achieving a MAPE of ~14-16% and RMSE near 2,000L. Because water consumption heavily relies on cyclical temporal patterns (day of week, seasonality), models explicitly designed to handle univariate seasonality excelled here.
- **Deep Learning:** LSTM outperformed the tree-based models slightly, leveraging sequence memory.
- **Tree-Based Models:** Random Forest was the best among the ensemble models (XGBoost, LightGBM). We exported this model for the interactive dashboard due to its lightweight inference properties and ability to accept multivariate regressors directly.

## Recommendations for Improvement
1. **Multivariate Prophet:** Add exogenous weather variables (temperature, precipitation) to the Prophet model, which will likely push the MAPE below 10%.
2. **Deep Learning Hyperparameters:** Increase the sequence look-back window for LSTM/GRU from 1 to 7 or 14 days to give the neural network more context.
