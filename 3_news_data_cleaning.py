import pandas as pd 
import numpy as np

# Read the CSV files
news_df = pd.read_csv(r"./news.csv")
stock_df = pd.read_csv(r"./stock_price.csv")

# Modify the date format for all rows at once in stock_df
stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d')

# Assuming the first column in news_df is the date column
date_column = news_df.columns[0]
news_df[date_column] = pd.to_datetime(news_df[date_column]).dt.strftime('%Y-%m-%d')

# Filter news_df to keep only dates that exist in stock_df
news_df = news_df[news_df[date_column].isin(stock_df['Date'])]

# Sort both dataframes by date
news_df = news_df.sort_values(date_column)
stock_df = stock_df.sort_values('Date')

# Save the filtered and sorted news_df
news_df.to_csv("news_data.csv", index=False)
print("Data cleaning complete. Filtered news data saved to news_data.csv")

# Print the first few rows and column names of both dataframes for debugging
print("\nFirst few rows of news_df:")
print(news_df.head())
print("\nColumn names of news_df:")
print(news_df.columns)

print("\nFirst few rows of stock_df:")
print(stock_df.head())
print("\nColumn names of stock_df:")
print(stock_df.columns)