import pickle
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# deeplearning modelling for crime prediction
df = pd.read_csv("exports/crime_df.csv", parse_dates=['Month'])

# Create binary target for burglary
df['target'] = df['Crime type'].str.lower().eq("burglary").astype(int)

# Time features
# Make sure 'Month' is datetime
df['Month'] = pd.to_datetime(df['Month'])

# Extract datetime components safely
df['hour'] = df['Month'].dt.hour
df['dayofweek'] = df['Month'].dt.dayofweek
df['month'] = df['Month'].dt.month


# Select features
features = ['Latitude', 'Longitude', 'hour', 'dayofweek', 'month']
X = df[features].fillna(0).values
y = df['target'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build binary classifier
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: probability of burglary
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

# Test input: location + time
sample = pd.DataFrame([{
    'Latitude': 40.7128,
    'Longitude': -74.0060,
    'hour': 15,
    'dayofweek': 2,  # Wednesday
    'month': 6
}])

sample_scaled = scaler.transform(sample)
prob = model.predict(sample_scaled)[0][0]

print(f"Predicted probability of burglary: {prob:.3f}")
print("Prediction:", "Burglary" if prob > 0.5 else "Not Burglary")

# time series prediction
# Filter to only burglary crimes
df['is_burglary'] = df['Crime type'].str.lower().eq("burglary").astype(int)

# Group by month and count burglaries
monthly_burglaries = df.groupby(df['Month'].dt.to_period("M"))['is_burglary'].sum().reset_index()
monthly_burglaries['Month'] = monthly_burglaries['Month'].dt.to_timestamp()
monthly_burglaries = monthly_burglaries.set_index('Month')

# Normalize the burglary counts
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(monthly_burglaries.values)

# Create time series sequences for LSTM
def create_sequences(data, window_size=12):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 12  # Using past 12 months to predict next month
X, y = create_sequences(scaled_series, window_size)

# Train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_split=0.1)

# Predict
preds = model.predict(X_test)
preds_inv = scaler.inverse_transform(preds)
y_test_inv = scaler.inverse_transform(y_test)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(preds_inv, label='Predicted')
plt.title("Burglary Count Forecast")
plt.xlabel("Months (test set)")
plt.ylabel("Number of Burglaries")
plt.legend()
plt.grid(True)
plt.show()

# # Initialize the Flask app
# app = Flask(__name__)

# #specify the file path in the directory
# file_path = 'crime.pkl'

# # Step 1: Load the model from the Pickle file when the app starts
# with open(file_path, 'rb') as f:
#     model = pickle.load(f)
#     print("Model loaded successfully:", model)

# @app.route('/')
# def home():
#     return "Welcome to the Crime prediction API!"

# # Step 2: Use the loaded model to make predictions in the /predict route
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input data from the POST request
#     data = request.get_json()
    
#     # Convert input data into a numpy array
#     input_data = np.array(data.get('input')).reshape(1, -1)
    
#      # Make a prediction using the loaded model
#     prediction = model.predict(input_data)[0]
    
#     # Return the prediction as a JSON response
#     return jsonify({'prediction': int(prediction)})









# if __name__ == '__main__':
#     app.run(debug=True)