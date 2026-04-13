import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_and_evaluate(X, y, target_name='heating_load'):
    print(f"\n{'='*50}")
    print(f"Training models for: {target_name}")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05,
                                 max_depth=4, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)

        results[name] = {
            'model': model,
            'mae':   mae,
            'rmse':  rmse,
            'preds': preds,
            'y_test': y_test
        }
        print(f"  {name:20s}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    # Pick best model
    best_name = min(results, key=lambda k: results[k]['mae'])
    print(f"\n  Best model: {best_name}")

    # Save best model to disk
    filename = f'models/best_{target_name}.pkl'
    joblib.dump(results[best_name]['model'], filename)
    print(f"  Saved → {filename}")

    return results, best_name, X_test, y_test