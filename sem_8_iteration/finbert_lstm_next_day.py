import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load sentiment analysis results
df_sentiment = pd.read_csv(r"sem_8_iteration\finbert_sentiment_results.csv")
df_stock = pd.read_csv(r"sem_8_iteration\nifty_50__stock_price.csv")

# Convert date columns to datetime
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'], format='%d-%m-%Y')
df_stock['Date'] = pd.to_datetime(df_stock['Date'], format='%Y-%m-%d')


# Merge dataframes on date
df = df_stock.merge(df_sentiment, left_on='Date', right_on='date')
df.drop(columns=['date'], inplace=True)

# Select features (sentiment scores) and target (next day's closing price)
features = ['daily_positive', 'daily_negative', 'daily_neutral']
X = df[features].values
y = df['Close'].shift(-1).dropna().values  # Predicting next day's closing price
X = X[:-1]  # Align dimensions

# Scale features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Reshape for LSTM
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Predict next day's stock movement
last_day_input = X_scaled[-1].reshape(1, 1, len(features))
predicted_price = model.predict(last_day_input)
predicted_price = scaler_y.inverse_transform(predicted_price)
print(f'Predicted next day closing price: {predicted_price[0][0]}')
