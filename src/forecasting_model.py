import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

class WaterDemandForecaster:
    def __init__(self, model_path="models/random_forest_model.pkl"):
        self.model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        self.model_path = model_path
        self.features = ['day_of_week', 'is_weekend', 'month', 'day', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
        self.target = 'consumption_liters'

    def train(self, train_df, val_df=None):
        X_train = train_df[self.features]
        y_train = train_df[self.target]
        
        self.model.fit(X_train, y_train)
        self.save_model()

    def predict(self, X):
        return self.model.predict(X[self.features])

    def evaluate(self, test_df):
        X_test = test_df[self.features]
        y_test = test_df[self.target]
        predictions = self.predict(test_df)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

if __name__ == "__main__":
    train_df = pd.read_csv("Dataset/processed/train.csv")
    val_df = pd.read_csv("Dataset/processed/val.csv")
    test_df = pd.read_csv("Dataset/processed/test.csv")
    
    forecaster = WaterDemandForecaster()
    forecaster.train(train_df, val_df)
    
    metrics = forecaster.evaluate(test_df)
    print("Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
