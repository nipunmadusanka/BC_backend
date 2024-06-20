from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the dictionary from the .pkl file
loaded_objects = joblib.load('mental_disease_predictor_with_encoder.pkl')

# Extract the model and label encoder from the dictionary
loaded_model = loaded_objects['model']
loaded_label_encoder = loaded_objects['label_encoder']

@app.route('/')
def index():
    return "Hello, Flask!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'No input data provided'}), 400

    features = data['features']
    
    # Ensure features is a list of lists
    if not isinstance(features, list) or not all(isinstance(i, list) for i in features):
        return jsonify({'error': 'Invalid input data format'}), 400

    try:
        # Make prediction
        predictions = loaded_model.predict(features)
        predicted_labels = loaded_label_encoder.inverse_transform(predictions)
        
        # Return predictions as JSON
        return jsonify({'predictions': predicted_labels.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check', methods=['GET'])
def check():
    new_data = [[3, 5, 4, 2, 3, 4, 2, 1, 5]]  # Example data
    try:
        # Make prediction using example data
        predictions = loaded_model.predict(new_data)
        predicted_labels = loaded_label_encoder.inverse_transform(predictions)
        
        # Return predictions as JSON
        return jsonify({'predictions': predicted_labels.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
