import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_predictions(y_test, preds, title='Actual vs Predicted'):
    fig = go.Figure()

    # Actual vs Predicted dots
    fig.add_trace(go.Scatter(
        x=list(y_test), y=list(preds),
        mode='markers',
        marker=dict(color='steelblue', size=6, opacity=0.7),
        name='Predictions'
    ))

    # Perfect prediction line
    min_val = min(min(y_test), min(preds))
    max_val = max(max(y_test), max(preds))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect prediction'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Actual',
        yaxis_title='Predicted',
        height=500
    )
    fig.show()


def plot_feature_importance(model, feature_names, title='Feature Importance'):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = abs(model.coef_)  

    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importances[indices]

    fig = go.Figure(go.Bar(
        x=sorted_importance,
        y=sorted_features,
        orientation='h',
        marker_color='steelblue'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        height=400
    )
    fig.show()


def plot_anomalies(df, column, anomalies, title='Anomaly Detection'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[~anomalies],
        y=df[column][~anomalies],
        mode='markers',
        marker=dict(color='steelblue', size=5),
        name='Normal'
    ))

    fig.add_trace(go.Scatter(
        x=df.index[anomalies],
        y=df[column][anomalies],
        mode='markers',
        marker=dict(color='red', size=9, symbol='x'),
        name='Anomaly'
    ))

    fig.update_layout(title=title, xaxis_title='Index',
                      yaxis_title=column, height=400)
    fig.show()