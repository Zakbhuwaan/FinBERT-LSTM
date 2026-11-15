import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# Performance metrics
models = ['MLP', 'LSTM', 'BERT-LSTM']
mae_values = [373.3453125, 69.53369140625, 104.8251953125]
mape_values = [0.01453783168323008, 0.0027602535415712243, 0.004060456080373899]
accuracy_values = [0.9854621683167699, 0.9854318882317031, 0.9959395439196261]

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Performance Comparison', fontsize=16)

# MAE plot
ax1.bar(models, mae_values)
ax1.set_title('Mean Absolute Error')
ax1.set_ylabel('MAE')
for i, v in enumerate(mae_values):
    ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom')

# MAPE plot
ax2.bar(models, mape_values)
ax2.set_title('Mean Absolute Percentage Error')
ax2.set_ylabel('MAPE')
for i, v in enumerate(mape_values):
    ax2.text(i, v, f'{v:.4f}', ha='center', va='bottom')

# Accuracy plot
ax3.bar(models, accuracy_values)
ax3.set_title('Accuracy')
ax3.set_ylabel('Accuracy')
ax3.set_ylim(0.98, 1.0)  # Adjust y-axis for better visibility
for i, v in enumerate(accuracy_values):
    ax3.text(i, v, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print analysis
print("Model Performance Analysis:")
print("---------------------------")
for i, model in enumerate(models):
    print(f"{model}:")
    print(f"  MAE:      {mae_values[i]:.2f}")
    print(f"  MAPE:     {mape_values[i]:.4f}")
    print(f"  Accuracy: {accuracy_values[i]:.4f}")
    print()

# Determine the best model for each metric
best_mae = min(mae_values)
best_mape = min(mape_values)
best_accuracy = max(accuracy_values)

print("Best performing models:")
print(f"  Lowest MAE:  {models[mae_values.index(best_mae)]} ({best_mae:.2f})")
print(f"  Lowest MAPE: {models[mape_values.index(best_mape)]} ({best_mape:.4f})")
print(f"  Highest Accuracy: {models[accuracy_values.index(best_accuracy)]} ({best_accuracy:.4f})")

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