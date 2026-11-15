from newsapi import NewsApiClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_news(year, month, day, newsapi):
    """Fetch top 10 India-based stock news headlines impacting NIFTY 50 for a specific day using NewsAPI."""
    try:
        date_str = f"{year}-{month:02d}-{day:02d}"
        articles = newsapi.get_everything(
            q='NIFTY OR stocks OR markets OR economy',
            from_param=date_str,
            to=date_str,
            language='en',
            sort_by='relevancy',
            page_size=10,
            domains=(
                "economictimes.indiatimes.com,"
                "moneycontrol.com,"
                "business-standard.com,"
                "livemint.com,"
                "financialexpress.com,"
                "thehindubusinessline.com"
            )
        )
        headlines = [article['title'].replace(',', '') for article in articles['articles']]
        headlines.extend([''] * (10 - len(headlines)))
        return headlines[:10]
    except Exception as e:
        print(f"Error fetching news for {year}-{month}-{day}: {e}")
        return [''] * 10

def generate_news_file():
    """
    Store NIFTY 50 related news headlines for the past month in a CSV file.
    """
    newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')  # Replace with your actual API key
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    mydates = pd.date_range(start=start_date, end=end_date)
    
    matrix = np.zeros((len(mydates) + 1, 11), dtype=object)
    matrix[0, 0] = "Date"
    for i in range(10):
        matrix[0, i + 1] = f"News {i + 1}"
    
    for i, single_date in enumerate(mydates):
        date_str = single_date.strftime("%Y-%m-%d")
        y, m, d = map(int, date_str.split("-"))
        print(f"Fetching news for {date_str}...")
        news_list = get_news(y, m, d, newsapi)
        matrix[i + 1, 0] = date_str
        for j, news in enumerate(news_list):
            matrix[i + 1, j + 1] = news
    
    df = pd.DataFrame(matrix)
    df.to_csv(r"standalone_lstm_model\nifty_50_stock_news.csv", index=False)
    print("News collection complete. Data saved to nifty_50_stock_news.csv")

generate_news_file()
