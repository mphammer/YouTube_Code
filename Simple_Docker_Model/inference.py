from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data
    data = request.json['data']
    data = np.array(data).reshape(1, -1)  # Reshape for prediction

    # Predict with the model
    prediction = model.predict(data)

    # Return prediction data
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5432)