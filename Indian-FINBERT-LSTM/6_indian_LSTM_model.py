import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

# hyperparameters
sequence_length = 3  # Reduced due to small dataset
epochs = 100
learning_rate = 0.01

# loading stock price data
stock_data = pd.read_csv("Indian-FINBERT-LSTM/indian_stock_price.csv")
column = ['Close']

len_stock_data = stock_data.shape[0]
print(f"Total data points: {len_stock_data}")

# Prepare data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

data = stock_data['Close'].values.reshape(-1, 1)
X, y = create_sequences(data, sequence_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# normalizing data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)

# Creating LSTM model
def model_create():
    tf.random.set_seed(1234)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(10, activation='relu', input_shape=(sequence_length, 1)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model

# prediction on test set
def predict(model):
    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# evaluation
def evaluate(predictions):
    y_test_inv = scaler.inverse_transform(y_test)
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