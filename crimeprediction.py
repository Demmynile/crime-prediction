import pickle
import numpy as np
from flask import Flask, request, jsonify
import crimeApis


# Initialize the Flask app
app = Flask(__name__)

#specify the file path in the directory
file_path = 'crime.pkl'

# Step 1: Load the model from the Pickle file when the app starts
with open(file_path, 'rb') as f:
    model = pickle.load(f)
    print("Model loaded successfully:", model)

@app.route('/')
def home():
    return "Welcome to the Crime prediction API!"

# Step 2: Use the loaded model to make predictions in the /predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.get_json()
    
    # Convert input data into a numpy array
    input_data = np.array(data.get('input')).reshape(1, -1)
    
     # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction)})









if __name__ == '__main__':
    app.run(debug=True)