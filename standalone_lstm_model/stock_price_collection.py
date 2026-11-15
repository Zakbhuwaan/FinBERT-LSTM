import pandas as pd
import yfinance as yf

def download_stock_data(ticker, start, end):

    #download stock price data from Yahoo Finance
   
    stock_data = yf.download(ticker, start, end)
    df = pd.DataFrame(stock_data)
    df.to_csv(r"standalone_lstm_model\nifty_50__stock_price.csv")


download_stock_data("^NSEI", "2025-01-03", "2025-02-02")