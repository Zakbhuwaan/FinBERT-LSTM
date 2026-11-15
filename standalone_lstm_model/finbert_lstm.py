import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('best_finbert_lstm_model.h5')

# Load the new dataset
new_sentiment = pd.read_csv(r"finbert_sentiment_results.csv")
new_stock = pd.read_csv(r"nifty_50__stock_price.csv")

# Convert date columns to datetime
new_sentiment['date'] = pd.to_datetime(new_sentiment['date'], format='%Y-%m-%d')
new_stock['Date'] = pd.to_datetime(new_stock['Date'], format='%Y-%m-%d')

# Merge dataframes on date
new_data = new_stock.merge(new_sentiment, left_on='Date', right_on='date')
new_data.drop(columns=['date'], inplace=True)

# Select features (sentiment scores) for prediction
features = ['daily_positive', 'daily_negative', 'daily_neutral']
X_new = new_data[features].values

# Scale the features using the same scaler that was used for training
scaler_X = MinMaxScaler()
X_new_scaled = scaler_X.fit_transform(X_new)

# Reshape for LSTM input (same as training data)
X_new_scaled = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

# Make predictions with the loaded model
predictions = model.predict(X_new_scaled)

# Inverse transform the predictions to get the original scale
scaler_y = MinMaxScaler()  # Initialize scaler for y (closing prices)
y_original = new_data['Close'].values.reshape(-1, 1)
scaler_y.fit(y_original)
predicted_prices = scaler_y.inverse_transform(predictions)

# Print the lengths to debug
print(f"Length of new_data: {len(new_data)}")
print(f"Length of predicted_prices: {len(predicted_prices)}")

# Generate trading signals (buy, hold, sell)
def generate_signal(current_price, predicted_price, threshold_buy=0.01, threshold_sell=0.01):
    percent_change = (predicted_price - current_price) / current_price * 100
    
    if percent_change >= threshold_buy:
        return "BUY", percent_change
    elif percent_change <= -threshold_sell:
        return "SELL", percent_change
    else:
        return "HOLD", percent_change

# Add signals to the data
signals = []
percentage_changes = []

# Make sure we don't go out of bounds
for i in range(len(predicted_prices)):
    if i < len(new_data):  # Safety check
        current_price = new_data['Close'].iloc[i]
        predicted_next_day = predicted_prices[i][0]
        signal, percent_change = generate_signal(current_price, predicted_next_day)
        signals.append(signal)
        percentage_changes.append(percent_change)

# Make sure all arrays have the same length as the DataFrame
new_data['Predicted_Next_Day'] = pd.Series([price[0] for price in predicted_prices], index=new_data.index[:len(predicted_prices)])
new_data['Signal'] = pd.Series(signals, index=new_data.index[:len(signals)])
new_data['Percent_Change'] = pd.Series(percentage_changes, index=new_data.index[:len(percentage_changes)])

# Display trading signals with context
print("\n===== TRADING SIGNALS =====")
for i in range(min(len(new_data), len(predicted_prices))):
    date = new_data['Date'].iloc[i].strftime('%Y-%m-%d')
    current = new_data['Close'].iloc[i]
    predicted = new_data['Predicted_Next_Day'].iloc[i]
    
    # Check if signal and percent_change exist for this row
    if i < len(signals):
        signal = signals[i]
        pct_change = percentage_changes[i]
        print(f"Date: {date} | Current: ${current:.2f} | Predicted: ${predicted:.2f} | Change: {pct_change:.2f}% | Signal: {signal}")
    else:
        print(f"Date: {date} | Current: ${current:.2f} | Predicted: ${predicted:.2f} | No signal available")

# Visualization of actual vs predicted prices with signals
plt.figure(figsize=(12, 6))
plt.plot(new_data['Date'], new_data['Close'], label='Actual Price', color='blue')

# Only plot available predictions
valid_indices = ~new_data['Predicted_Next_Day'].isna()
plt.plot(new_data['Date'][valid_indices], 
         new_data['Predicted_Next_Day'][valid_indices], 
         label='Predicted Next Day', color='green', linestyle='--')

# Plot buy/sell signals
buy_signals = new_data[new_data['Signal'] == 'BUY']
sell_signals = new_data[new_data['Signal'] == 'SELL']

if not buy_signals.empty:
    plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', s=100, label='Buy Signal')
if not sell_signals.empty:
    plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', s=100, label='Sell Signal')

plt.title('Stock Price Prediction with Trading Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('trading_signals.png')
plt.show()

# Save the results to a CSV file
new_data.to_csv('trading_signals.csv', index=False)
print("\nTrading signals saved to 'trading_signals.csv'")

