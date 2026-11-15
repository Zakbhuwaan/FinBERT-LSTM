import csv
import time
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from datetime import datetime, timedelta

def get_news(year, month, day, newsapi):
    """
    Get top 10 most relevant Indian finance news headlines for a specific day using NewsAPI.
    """
    try:
        # Format the date
        date_str = f"{year}-{month:02d}-{day:02d}"
        
        # Fetch news articles for the specified date
        articles = newsapi.get_everything(
            q='finance OR stocks OR market',
            from_param=date_str,
            to=date_str,
            language='en',
            sort_by='relevancy',
            page_size=10,  # Get top 10 news
            domains='economictimes.indiatimes.com,moneycontrol.com,financialexpress.com,trendline,tijorifinance.in,screener.in' #trendline, tijorifinance.in, screener.in, Webcrawling  NSE Blog, BSE Blog
        )
        
        # Extract and return article headlines
        headlines = [article['title'].replace(',', '') for article in articles['articles']]
        
        # If less than 10 headlines, pad with empty strings
        headlines.extend([''] * (10 - len(headlines)))
        
        return headlines[:10]  # Ensure we return exactly 10 headlines
    except Exception as e:
        print(f"Error fetching news for {year}-{month}-{day}: {e}")
        return [''] * 10  # Return 10 empty strings in case of error

def generate_news_file():
    """
    Store news headlines every day of the past 1 month in a CSV file.
    """
    # Initialize the NewsAPI client
    newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')  # Replace with your actual API key
    
    # Calculate date range for the past 1 month
    end_date = datetime.today()  # Current date
    start_date = end_date - timedelta(days=30)  # 30 days ago
    
    mydates = pd.date_range(start=start_date, end=end_date)
    
    # Create a matrix to store the headlines, with space for 10 news headlines each day
    matrix = np.zeros((len(mydates) + 1, 11), dtype=object)
    matrix[0, 0] = "Date"  # Header for the date
    for i in range(10):
        matrix[0, i + 1] = f"News {i + 1}"  # Headers for each news item
    
    # Fetch news for each date within the past 1 month
    for i, single_date in enumerate(mydates):
        date_str = single_date.strftime("%Y-%m-%d")
        y, m, d = map(int, date_str.split("-"))
        print(f"Fetching news for {date_str}...")  # Debugging output
        news_list = get_news(y, m, d, newsapi)  # Fetch the news headlines
        matrix[i + 1, 0] = date_str  # Store the date in the matrix
        for j, news in enumerate(news_list):
            matrix[i + 1, j + 1] = news  # Store the headlines
        
        # Add a delay to avoid exceeding API rate limits
        time.sleep(1)
    
    # Convert the matrix to a DataFrame and save it as a CSV file
    df = pd.DataFrame(matrix)
    df.to_csv("indian_finance_news_past_month.csv", index=False)
    print("News collection complete. Data saved to indian_finance_news_past_month.csv")

generate_news_file()