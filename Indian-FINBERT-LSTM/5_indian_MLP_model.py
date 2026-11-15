import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

# loading stock price data
stock_data = pd.read_csv("Indian-FINBERT-LSTM/indian_stock_price.csv")
column = ['Close']

len_stock_data = stock_data.shape[0]
print(f"Total data points: {len_stock_data}")

# Prepare data
X = stock_data['Close'].values[:-1].reshape(-1, 1)
y = stock_data['Close'].values[1:].reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# normalizing data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# Creating simple model
def model_create():
    tf.random.set_seed(1234)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model

# prediction on test set
def predict(model):
    predictions = model.predict(X_test, verbose=0)
    predictions = y_scaler.inverse_transform(predictions)
    return predictions

# evaluation
def evaluate(predictions):
    y_test_inv = y_scaler.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_inv, predictions)
    mape = mean_absolute_percentage_error(y_test_inv, predictions)
    return mae, mape, (1 - mape)

# trial runs
def run_model(n):
    total_mae = total_mape = total_acc = 0
    for i in range(n):
        model = model_create()
        predictions = predict(model)
        mae, mape, acc = evaluate(predictions)
        total_mae += mae
        total_mape += mape 
        total_acc += acc 
    return (total_mae / n), (total_mape / n), (total_acc / n), predictions.tolist()

mae, mape, acc, preds = run_model(1)

print(f"Mean Absolute Error = {mae}")
print(f"Mean Absolute Percentage Error = {mape}%")
print(f"Accuracy = {acc}")