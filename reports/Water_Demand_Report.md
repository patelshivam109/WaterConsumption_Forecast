# Water Demand & Utility Monitoring Report

## Executive Summary
This report summarizes the operational insights derived from the Water Consumption Forecasting and Spike Detection System. By analyzing 180 days of water usage data across 5 distinct regions, we have successfully implemented a predictive and monitoring infrastructure to assist utility providers in optimizing water distribution.

## Demand Categorization
To simplify operational decision-making, water consumption was statistically categorized into three bands based on historical percentiles across the regions:
- **Low Demand:** < 33rd Percentile (~9,187 Liters or less)
- **Medium Demand:** 33rd - 67th Percentile
- **High Demand:** > 67th Percentile (~16,365 Liters or more)
These categories allow utility managers to quickly assess current consumption rates and adjust pumping operations or resource allocations accordingly.

## Anomaly and Spike Detection System
A robust statistical anomaly detector using dynamic rolling Z-scores (7-day window, 2.0 standard deviations threshold) is actively monitoring usage. 
- **Spike Alerts:** The system correctly identifies sudden surges in demand. These may represent unexpected industrial usage, large-scale leakages, or system bursts requiring immediate maintenance checks.
- **Drop Alerts:** Sudden, unexpected drops in usage are similarly tracked, which might point to infrastructure blockage or severe metering faults.
- Overall, approximately **71 distinct anomalies** were detected in the historical timeframe, giving utility providers concrete historical events to cross-reference with known infrastructural incidents.

## Actionable Outcomes & Next Steps
1. **Utility Dashboard Deployment:** The operational dashboard built with Streamlit is ready to be deployed. Utility managers can use this to visually track consumption and view real-time Early Warning Alerts.
2. **Infrastructure Triage:** Regions displaying high frequencies of "Spikes" should be prioritized for infrastructure inspection to mitigate potential chronic leakages.
3. **Capacity Planning:** Utilize the forecasted outputs (from the XGBoost regression module) to inform daily capacity requirements, minimizing unnecessary pumping energy during forecasted low-demand windows.
