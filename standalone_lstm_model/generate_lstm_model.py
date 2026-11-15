import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
# LOAD DATASETS
def load_datasets():
    stock_df = pd.read_csv("nifty_50__stock_price.csv", parse_dates=["Date"]).sort_values("Date")
    sentiment_df = pd.read_csv("finbert_sentiment_results.csv", parse_dates=["Date"]).sort_values("Date")
    return stock_df["Close"].values.reshape(-1, 1), sentiment_df["Sentiment_Score"].values.reshape(-1, 1)
# PREPARE DATASET FOR LSTM
def prepare_dataset(stock_prices, sentiment_scores, lookback=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_scaled = scaler.fit_transform(stock_prices)
    combined_data = np.hstack((stock_scaled, sentiment_scores))

    X, y = [], []
    for i in range(len(combined_data) - lookback):
        X.append(combined_data[i:i+lookback])
        y.append(stock_scaled[i+lookback])

    return np.array(X), np.array(y), scaler
# BUILD LSTM MODEL
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# TRAIN OR LOAD MODEL
def train_or_load_lstm(X_train, y_train, X_test, y_test, lookback):
    model_path = "finbert_lstm_stock_model.h5"
    if os.path.exists(model_path):
        print("Loaded pre-trained model.")
        return load_model(model_path)

    print("Training new LSTM model...")
    model = build_lstm_model((lookback, X_train.shape[2]))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ]
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, callbacks=callbacks)
    return model
# MAIN EXECUTION
if __name__ == "__main__":
    lookback = 10
    stock_prices, sentiment_scores = load_datasets()
    X, y, scaler = prepare_dataset(stock_prices, sentiment_scores, lookback)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = train_or_load_lstm(X_train, y_train, X_test, y_test, lookback)
    # PREDICT & VALIDATE
    predictions = scaler.inverse_transform(model.predict(X_test))
    actual_prices = scaler.inverse_transform(y_test)

    mse = np.mean((predictions - actual_prices) ** 2)
    print(f"Prediction Error (MSE): {mse:.4f}")
    # SAVE MODEL IF ERROR IS MINIMAL
    if mse < 0.05:
        model.save("finbert_lstm_stock_model.h5")
        print("Model saved as 'finbert_lstm_stock_model.h5'.")
    else:
        print("Model not saved. Prediction error too high.")
