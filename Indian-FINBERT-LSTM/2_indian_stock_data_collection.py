import pandas as pd

def download_stock_data(ticker, start, end):
    """
    download stock price data from Yahoo Finance
    """
    import yfinance as yf
    stock_data = yf.download(ticker, start, end)
    df = pd.DataFrame(stock_data)
    df.to_csv("indian_stock_price.csv")


download_stock_data("^NSEI", "2024-08-29", "2024-09-28")