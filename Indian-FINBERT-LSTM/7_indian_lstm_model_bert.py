import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

# hyperparameters
sequence_length = 2  # Reduced dataset
epochs = 100
learning_rate = 0.01

# loading stock price and news data
stock_data = pd.read_csv("Indian-FINBERT-LSTM/indian_stock_price.csv")
news_data = pd.read_csv("Indian-FINBERT-LSTM/indian_sentiment.csv")

# Ensure both dataframes have the same length
min_length = min(len(stock_data), len(news_data))
stock_data = stock_data.iloc[:min_length]
news_data = news_data.iloc[:min_length]

print(f"Total data points: {len(stock_data)}")

# Prepare data
def create_sequences(price_data, sentiment_data, seq_length):
    X, y = [], []
    for i in range(len(price_data) - seq_length):
        X.append(np.column_stack((price_data[i:(i + seq_length)], 
                                  sentiment_data[i:(i + seq_length)])))
        y.append(price_data[i + seq_length])
    return np.array(X), np.array(y)

price_data = stock_data['Close'].values.reshape(-1, 1)
sentiment_data = news_data['FinBERT score'].values.reshape(-1, 1)

X, y = create_sequences(price_data, sentiment_data, sequence_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# normalizing data
price_scaler = MinMaxScaler()
sentiment_scaler = MinMaxScaler()

X_train[:,:,0] = price_scaler.fit_transform(X_train[:,:,0].reshape(-1, 1)).reshape(X_train[:,:,0].shape)
X_train[:,:,1] = sentiment_scaler.fit_transform(X_train[:,:,1].reshape(-1, 1)).reshape(X_train[:,:,1].shape)

X_test[:,:,0] = price_scaler.transform(X_test[:,:,0].reshape(-1, 1)).reshape(X_test[:,:,0].shape)
X_test[:,:,1] = sentiment_scaler.transform(X_test[:,:,1].reshape(-1, 1)).reshape(X_test[:,:,1].shape)

y_train = price_scaler.transform(y_train)
y_test = price_scaler.transform(y_test)

# Creating LSTM model
def model_create():
    tf.random.set_seed(1234)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(10, activation='relu', input_shape=(sequence_length, 2)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model

# prediction on test set
def predict(model):
    predictions = model.predict(X_test, verbose=0)
    predictions = price_scaler.inverse_transform(predictions)
    return predictions

# evaluation
def evaluate(predictions):
    y_test_inv = price_scaler.inverse_transform(y_test)
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
print(f"Mean Absolute Percentage Error = {mape}")
print(f"Accuracy = {acc}")