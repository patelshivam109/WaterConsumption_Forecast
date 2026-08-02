# Forecast Evaluation Report

## Overview
This report details the performance of the XGBoost Regressor model trained to forecast daily water demand. The model utilizes time-series features (lags and rolling averages) and temporal features (day of week, is_weekend, month, day).

## Model Configuration
- **Algorithm:** XGBoost Regressor
- **Objective:** `reg:squarederror`
- **Hyperparameters:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`
- **Features Used:** `day_of_week`, `is_weekend`, `month`, `day`, `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_std_7`

## Evaluation Metrics (Test Set)
The dataset was split chronologically (70% Train, 15% Validation, 15% Test). The baseline model achieved the following performance on the test set:
- **Root Mean Squared Error (RMSE):** ~4449.57 Liters
- **Mean Absolute Error (MAE):** ~3611.60 Liters
- **Mean Absolute Percentage Error (MAPE):** ~36.00%
- **R² Score:** Negative (-0.18)

## Analysis of Results
The relatively high MAPE and negative R² score indicate that the baseline model struggles to accurately generalize the forecasting strictly based on a 7-day rolling window and simple lag features. 

## Recommendations for Improvement
1. **Hyperparameter Tuning:** Conduct a Grid Search or Random Search to find optimal parameters (e.g., deeper trees, different learning rates, adjusting `subsample` and `colsample_bytree`).
2. **Additional Features:** Incorporate external regressors such as historical weather data (temperature, rainfall), which significantly drive water demand.
3. **Alternative Algorithms:** Compare the XGBoost baseline against traditional time-series models like SARIMA or Prophet, which natively handle seasonal components better.
