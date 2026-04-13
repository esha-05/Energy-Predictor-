import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_clean(filepath):
    df = pd.read_excel(filepath)

    # Rename columns so they're human-readable
    df = df.rename(columns={
        'X1': 'compactness',
        'X2': 'surface_area',
        'X3': 'wall_area',
        'X4': 'roof_area',
        'X5': 'height',
        'X6': 'orientation',
        'X7': 'glazing_area',
        'X8': 'glazing_distribution',
        'Y1': 'heating_load',
        'Y2': 'cooling_load'
    })

    # Check for missing values
    print("Missing values:\n", df.isnull().sum())

    # Remove duplicates if any
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Dataset shape after cleaning: {df.shape}")
    return df


def get_features_and_targets(df):
    # These are our input columns (what the model sees)
    feature_cols = [
        'compactness', 'surface_area', 'wall_area', 'roof_area',
        'height', 'orientation', 'glazing_area', 'glazing_distribution'
    ]

    X = df[feature_cols]
    y_heating = df['heating_load']  
    y_cooling = df['cooling_load']  

    return X, y_heating, y_cooling


def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)   
    return X_train_scaled, X_test_scaled, scaler