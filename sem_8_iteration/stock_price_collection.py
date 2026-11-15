import pandas as pd

def download_stock_data(ticker, start, end):
    """
    download stock price data from Yahoo Finance
    """
    import yfinance as yf
    stock_data = yf.download(ticker, start, end)
    df = pd.DataFrame(stock_data)
    df.to_csv("nifty_50__stock_price.csv")


download_stock_data("^NSEI", "2024-12-21", "2025-01-21")