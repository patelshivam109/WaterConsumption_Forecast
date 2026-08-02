import pandas as pd
import numpy as np

class AnomalyDetector:
    def __init__(self, threshold_std=2.0):
        self.threshold_std = threshold_std

    def detect_anomalies(self, df):
        """
        Detects anomalies where consumption deviates significantly from the rolling mean.
        Uses historical actual data (rolling_mean_7 and rolling_std_7).
        """
        df = df.copy()
        
        # Calculate z-score based on the rolling window
        # To avoid division by zero
        rolling_std = df['rolling_std_7'].replace(0, np.nan)
        df['z_score'] = (df['consumption_liters'] - df['rolling_mean_7']) / rolling_std
        
        df['is_anomaly'] = (df['z_score'].abs() > self.threshold_std).astype(int)
        
        # Determine if it's a spike or drop
        df['anomaly_type'] = 'Normal'
        df.loc[(df['is_anomaly'] == 1) & (df['z_score'] > 0), 'anomaly_type'] = 'Spike'
        df.loc[(df['is_anomaly'] == 1) & (df['z_score'] < 0), 'anomaly_type'] = 'Drop'
        
        return df
        
    def categorize_demand(self, df):
        """
        Categorize demand into Low, Medium, and High based on percentiles.
        """
        df = df.copy()
        
        low_threshold = df['consumption_liters'].quantile(0.33)
        high_threshold = df['consumption_liters'].quantile(0.67)
        
        def get_category(val):
            if val < low_threshold:
                return 'Low'
            elif val > high_threshold:
                return 'High'
            else:
                return 'Medium'
                
        df['demand_category'] = df['consumption_liters'].apply(get_category)
        return df

if __name__ == "__main__":
    df = pd.read_csv("Dataset/processed/water_consumption_processed.csv")
    detector = AnomalyDetector()
    df_anomalies = detector.detect_anomalies(df)
    df_categorized = detector.categorize_demand(df_anomalies)
    
    anomalies_count = df_categorized['is_anomaly'].sum()
    print(f"Total Anomalies Detected: {anomalies_count}")
    print("\nDemand Category Distribution:")
    print(df_categorized['demand_category'].value_counts())
