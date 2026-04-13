import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app) 

# Load both trained models
model_heating = joblib.load('models/best_heating_load.pkl')
model_cooling = joblib.load('models/best_cooling_load.pkl')

# Feature order must match training exactly
FEATURE_ORDER = [
    'compactness', 'surface_area', 'wall_area', 'roof_area',
    'height', 'orientation', 'glazing_area', 'glazing_distribution'
]

@app.route('/')
def home():
    return jsonify({
        'project': 'Smart Energy Predictor',
        'endpoints': {
            'POST /predict': 'Send building features, get heating/cooling load prediction',
            'GET /health':   'Check if server is running'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = np.array([data[f] for f in FEATURE_ORDER]).reshape(1, -1)
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400

    heating = float(model_heating.predict(features)[0])
    cooling = float(model_cooling.predict(features)[0])

    return jsonify({
        'heating_load': round(heating, 3),
        'cooling_load': round(cooling, 3),
        'unit': 'kWh/m²'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)