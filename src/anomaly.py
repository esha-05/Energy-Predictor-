import pandas as pd
import numpy as np


def detect_anomalies_iqr(series: pd.Series, factor=1.5):
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    anomalies = (series < lower) | (series > upper)
    print(f"IQR method → {anomalies.sum()} anomalies found "
          f"(lower={lower:.2f}, upper={upper:.2f})")
    return anomalies


def detect_anomalies_zscore(series: pd.Series, threshold=2.5):
    z = (series - series.mean()) / series.std()
    anomalies = abs(z) > threshold
    print(f"Z-score method → {anomalies.sum()} anomalies found")
    return anomalies