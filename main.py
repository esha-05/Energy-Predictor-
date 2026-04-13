from src.preprocess import load_and_clean, get_features_and_targets, scale_features
from src.train import train_and_evaluate
from src.anomaly import detect_anomalies_iqr
from src.visualize import plot_predictions, plot_feature_importance, plot_anomalies

FEATURE_NAMES = [
    'compactness', 'surface_area', 'wall_area', 'roof_area',
    'height', 'orientation', 'glazing_area', 'glazing_distribution'
]

# STEP 1: Load and clean
df = load_and_clean('data/ENB2012_data.xlsx')

# STEP 2: Split into features and targets
X, y_heating, y_cooling = get_features_and_targets(df)

# STEP 3: Scale features
X_train_s, X_test_s, scaler = scale_features(
    X.iloc[:int(len(X)*0.8)],
    X.iloc[int(len(X)*0.8):]
)

# STEP 4: Train for Heating Load
results_h, best_h, X_test_h, y_test_h = train_and_evaluate(X, y_heating, 'heating_load')

# STEP 5: Train for Cooling Load
results_c, best_c, X_test_c, y_test_c = train_and_evaluate(X, y_cooling, 'cooling_load')

# STEP 6: Visualize predictions
plot_predictions(y_test_h, results_h[best_h]['preds'], 'Heating Load — Actual vs Predicted')
plot_predictions(y_test_c, results_c[best_c]['preds'], 'Cooling Load — Actual vs Predicted')

# STEP 7: Feature importance
plot_feature_importance(results_h[best_h]['model'], FEATURE_NAMES, 'Heating Load — Feature Importance')
plot_feature_importance(results_c[best_c]['model'], FEATURE_NAMES, 'Cooling Load — Feature Importance')

# STEP 8: Anomaly detection on targets
anomalies_h = detect_anomalies_iqr(df['heating_load'])
anomalies_c = detect_anomalies_iqr(df['cooling_load'])

plot_anomalies(df, 'heating_load', anomalies_h, 'Heating Load Anomalies')
plot_anomalies(df, 'cooling_load', anomalies_c, 'Cooling Load Anomalies')