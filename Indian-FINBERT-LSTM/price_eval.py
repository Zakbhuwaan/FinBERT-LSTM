# Add these imports at the beginning of the file
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# Add this after the existing code

# Load the data
test = pd.read_csv("Indian-FINBERT-LSTM/indian_stock_price.csv")['Close'].values[-len(mae_values):]
mlp_preds = pd.read_csv("Indian-FINBERT-LSTM/mlp_predictions.csv")['Prediction'].values
lstm_preds = pd.read_csv("Indian-FINBERT-LSTM/lstm_predictions.csv")['Prediction'].values
bert_lstm_preds = pd.read_csv("Indian-FINBERT-LSTM/bert_lstm_predictions.csv")['Prediction'].values

# Calculate RMSE for each model
rmse_mlp = sqrt(mean_squared_error(test, mlp_preds))
rmse_lstm = sqrt(mean_squared_error(test, lstm_preds))
rmse_bert_lstm = sqrt(mean_squared_error(test, bert_lstm_preds))

# Create a new figure for price comparison
plt.figure(figsize=(12, 6))
plt.plot(test, label='Actual', linewidth=2)
plt.plot(mlp_preds, label='MLP', linestyle='--')
plt.plot(lstm_preds, label='LSTM', linestyle='--')
plt.plot(bert_lstm_preds, label='BERT-LSTM', linestyle='--')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.show()

# Print RMSE analysis
print("\nRoot Mean Squared Error (RMSE) Analysis:")
print("----------------------------------------")
print(f"MLP RMSE:      {rmse_mlp:.2f}")
print(f"LSTM RMSE:     {rmse_lstm:.2f}")
print(f"BERT-LSTM RMSE: {rmse_bert_lstm:.2f}")

# Determine the best model based on RMSE
best_rmse = min(rmse_mlp, rmse_lstm, rmse_bert_lstm)
best_model = ['MLP', 'LSTM', 'BERT-LSTM'][[rmse_mlp, rmse_lstm, rmse_bert_lstm].index(best_rmse)]

print(f"\nBest performing model based on RMSE: {best_model} (RMSE: {best_rmse:.2f})")