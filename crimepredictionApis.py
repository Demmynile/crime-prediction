import pickle
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , LSTM
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



# Crime Prediction API using Flask
# This API provides endpoints for predicting burglary using both machine learning and deep learning models,

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Crime prediction API!"

# this is for predicting burglary using machine learning(random forest)
# Load the model
try:
    with open('crime.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.route('/api/predictions/predict_ml_crime', methods=['POST'])
def predictML():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        data = request.get_json()

        # Create a DataFrame with the expected features
        
        valid_features = model.feature_names_in_
        sample = pd.DataFrame([{k: data[k] for k in valid_features}])


        # Predict using the loaded model
        prob = float(model.predict(sample)[0])
        result = "Burglary" if prob > 0.5 else "Not Burglary"

        return jsonify({
            'probability': round(prob, 3),
            'prediction': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
# this is for predicting burglary using deep learning
@app.route('/api/predictions/predict_deep_crime', methods=['POST'])
def predictDeep():
        # Load and preprocess data
    df = pd.read_csv("exports/crime_df.csv", parse_dates=['Month'])
    df['target'] = df['Crime type'].str.lower().eq("burglary").astype(int)
    df['Month'] = pd.to_datetime(df['Month'])

    df['hour'] = df['Month'].dt.hour
    df['dayofweek'] = df['Month'].dt.dayofweek
    df['month'] = df['Month'].dt.month

    features = ['Latitude', 'Longitude', 'hour', 'dayofweek', 'month']
    X = df[features].fillna(0).values
    y = df['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define and train model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)
    data = request.get_json()

    try:
        sample = pd.DataFrame([{
            'Latitude': data['Latitude'],
            'Longitude': data['Longitude'],
            'hour': data['hour'],
            'dayofweek': data['dayofweek'],
            'month': data['month']
        }])

        sample_scaled = scaler.transform(sample)
        prob = float(model.predict(sample_scaled)[0][0])
        result = "Burglary" if prob > 0.5 else "Not Burglary"

        return jsonify({
            'probability': round(prob, 3),
            'prediction': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400



# for forcasting
def create_sequences(data, window_size=12):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

@app.route('/api/predictions/forecast', methods=['POST'])
def forecast_burglary():
    # Load CSV file
    df = pd.read_csv("exports/crime_df.csv", parse_dates=['Month'])

    # Parse datetime and preprocess
    df['Month'] = pd.to_datetime(df['Month'])
    df['is_burglary'] = df['Crime type'].str.lower().eq("burglary").astype(int)
    
    # Group and normalize
    monthly_burglaries = df.groupby(df['Month'].dt.to_period("M"))['is_burglary'].sum().reset_index()
    monthly_burglaries['Month'] = monthly_burglaries['Month'].dt.to_timestamp()
    monthly_burglaries = monthly_burglaries.set_index('Month')

    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(monthly_burglaries.values)

    window_size = 12
    X, y = create_sequences(scaled_series, window_size)

    if len(X) < 1:
        return jsonify({'error': 'Not enough data to create sequences'}), 400

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # LSTM Model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=8, validation_split=0.1, verbose=0)

    preds = model.predict(X_test)
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test)

    result = {
        'actual': y_test_inv.flatten().tolist(),
        'predicted': preds_inv.flatten().tolist()
    }

    return jsonify(result)





if __name__ == '__main__':
    app.run(debug=True)