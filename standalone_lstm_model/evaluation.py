import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Load actual and predicted stock prices
actual_prices = pd.read_csv(r"standalone_lstm_model\nifty_50__stock_price.csv")['Close']
predicted_prices = pd.read_csv(r"standalone_lstm_model\predicted_stock_prices.csv")['Prediction']

# Calculate evaluation metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
accuracy = 1 - mape

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Accuracy: {accuracy}")