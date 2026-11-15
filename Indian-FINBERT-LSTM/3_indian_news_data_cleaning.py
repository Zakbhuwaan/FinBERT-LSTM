import pandas as pd 
import numpy as np

news_df = pd.read_csv(r"Indian-FINBERT-LSTM/indian_finance_news_past_month.csv")
stock_df = pd.read_csv(r"Indian-FINBERT-LSTM/indian_stock_price.csv")

# Modify the date format for all rows at once
stock_df['Date'] = stock_df['Date'].str[:10]

news_df = news_df[news_df['0'].isin(stock_df['Date'].tolist())]

news_df.to_csv("indian_news_data.csv", index=False)