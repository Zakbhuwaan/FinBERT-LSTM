import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load data
stock_data = pd.read_csv("Indian-FINBERT-LSTM/indian_stock_price.csv")
news_data = pd.read_csv("Indian-FINBERT-LSTM/indian_sentiment.csv")

# Ensure both dataframes have the same length
min_length = min(len(stock_data), len(news_data))
stock_data = stock_data.iloc[:min_length]
news_data = news_data.iloc[:min_length]

# Prepare data for MLP and LSTM
X = stock_data['Close'].values[:-1].reshape(-1, 1)
y = stock_data['Close'].values[1:].reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.transform(y_train)
y_test_scaled = scaler.transform(y_test)

# MLP model
def create_mlp_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# LSTM model
def create_lstm_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(10, activation='relu', input_shape=(3, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

# BERT-LSTM model
def create_bert_lstm_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(10, activation='relu', input_shape=(2, 2)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

# Train and predict with MLP
mlp_model = create_mlp_model()
mlp_model.fit(X_train_scaled, y_train_scaled, epochs=100, verbose=0)
mlp_preds = scaler.inverse_transform(mlp_model.predict(X_test_scaled))

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(stock_data['Close'].values.reshape(-1, 1), 3)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Normalize LSTM data
X_train_lstm_scaled = scaler.fit_transform(X_train_lstm.reshape(-1, 1)).reshape(X_train_lstm.shape)
X_test_lstm_scaled = scaler.transform(X_test_lstm.reshape(-1, 1)).reshape(X_test_lstm.shape)
y_train_lstm_scaled = scaler.transform(y_train_lstm)
y_test_lstm_scaled = scaler.transform(y_test_lstm)

# Train and predict with LSTM
lstm_model = create_lstm_model()
lstm_model.fit(X_train_lstm_scaled, y_train_lstm_scaled, epochs=100, verbose=0)
lstm_preds = scaler.inverse_transform(lstm_model.predict(X_test_lstm_scaled))

# Prepare data for BERT-LSTM
def create_bert_sequences(price_data, sentiment_data, seq_length):
    X, y = [], []
    for i in range(len(price_data) - seq_length):
        X.append(np.column_stack((price_data[i:(i + seq_length)], 
                                  sentiment_data[i:(i + seq_length)])))
        y.append(price_data[i + seq_length])
    return np.array(X), np.array(y)

price_data = stock_data['Close'].values.reshape(-1, 1)
sentiment_data = news_data['FinBERT score'].values.reshape(-1, 1)

X_bert, y_bert = create_bert_sequences(price_data, sentiment_data, 2)
X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(X_bert, y_bert, test_size=0.2, random_state=42)

# Normalize BERT-LSTM data
price_scaler = MinMaxScaler()
sentiment_scaler = MinMaxScaler()

X_train_bert[:,:,0] = price_scaler.fit_transform(X_train_bert[:,:,0].reshape(-1, 1)).reshape(X_train_bert[:,:,0].shape)
X_train_bert[:,:,1] = sentiment_scaler.fit_transform(X_train_bert[:,:,1].reshape(-1, 1)).reshape(X_train_bert[:,:,1].shape)

X_test_bert[:,:,0] = price_scaler.transform(X_test_bert[:,:,0].reshape(-1, 1)).reshape(X_test_bert[:,:,0].shape)
X_test_bert[:,:,1] = sentiment_scaler.transform(X_test_bert[:,:,1].reshape(-1, 1)).reshape(X_test_bert[:,:,1].shape)

y_train_bert_scaled = price_scaler.transform(y_train_bert)
y_test_bert_scaled = price_scaler.transform(y_test_bert)

# Train and predict with BERT-LSTM
bert_lstm_model = create_bert_lstm_model()
bert_lstm_model.fit(X_train_bert, y_train_bert_scaled, epochs=100, verbose=0)
bert_lstm_preds = price_scaler.inverse_transform(bert_lstm_model.predict(X_test_bert))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', linewidth=2)
plt.plot(mlp_preds, label='MLP', linestyle='--')
plt.plot(lstm_preds, label='LSTM', linestyle='--')
plt.plot(bert_lstm_preds, label='BERT-LSTM', linestyle='--')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.show()