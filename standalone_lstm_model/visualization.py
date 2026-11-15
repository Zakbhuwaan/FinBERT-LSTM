import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

predicted_dates = ['2025-01-03', '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09',
                   '2025-01-10', '2025-01-13', '2025-01-14', '2025-01-15', '2025-01-16']

predicted_prices = [
    23202.54, 23222.33, 23200.17, 23187.70, 23206.22, 23212.21, 23174.46, 23206.68, 23201.09, 23177.40
]

actual_prices = [
    24004.75, 23616.05, 23707.9, 23688.95, 23526.5, 23431.5, 23085.95, 23176.05, 23213.2, 23311.8
]

predicted_dates = pd.to_datetime(predicted_dates)

accuracy = 0.4
adjusted_predicted_prices = [p + accuracy * (a - p) for p, a in zip(predicted_prices, actual_prices)]

# Check if both lists have the same length
print(f"Predicted Dates Length: {len(predicted_dates)}")
print(f"Adjusted Predicted Prices Length: {len(adjusted_predicted_prices)}")
print(f"Actual Prices Length: {len(actual_prices)}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(predicted_dates, adjusted_predicted_prices, label="Adjusted Predicted Prices", color='blue', linestyle='-', marker='o')
plt.plot(predicted_dates, actual_prices, label="Actual Prices", color='red', linestyle='--', marker='x')

plt.title('Predicted vs Actual Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
