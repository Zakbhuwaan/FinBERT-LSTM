import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt

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

# Define model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    
    # Tune the number of LSTM units in the first layer
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=32), 
                   return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    
    # Add dropout layers with tunable dropout rate
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Second LSTM layer with tunable units
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), return_sequences=False))
    
    # Add dropout
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Dense layers with tunable units
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    model.add(Dense(1))
    
    # Compile model with Adam optimizer and mean squared error loss
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']), loss='mean_squared_error')
    
    return model

# Initialize the Keras Tuner object
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    hyperband_iterations=2,
    directory='hyperband_tuning',
    project_name='finbert_lstm_tuning'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters
print("Best Hyperparameters:", best_hyperparameters.values)

# Evaluate the best model
loss = best_model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Predict using the best model
last_day_input = X_scaled[-1].reshape(1, 1, len(features))
predicted_price = best_model.predict(last_day_input)
predicted_price = scaler_y.inverse_transform(predicted_price)
print(f'Predicted next day closing price: {predicted_price[0][0]}')

# Get the last date in the dataset to predict for the next day
prediction_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
print(f"Predicted date: {prediction_date.strftime('%Y-%m-%d')}")

# Save the best model
best_model.save('best_finbert_lstm_model.h5')
print("Model saved as 'best_finbert_lstm_model.h5'")
