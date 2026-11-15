import pandas as pd 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

def FinBERT_sentiment_score(headlines):
    """
    Compute sentiment score using pretrained FinBERT on -1 to 1 scale. -1 being negative and 1 being positive
    """
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    
    combined_headline = ' '.join([str(h) for h in headlines if pd.notna(h) and h != '0'])
    result = nlp(combined_headline)
    
    if result[0]['label'] == "positive":
        return result[0]['score']
    elif result[0]['label'] == "neutral":
        return 0
    else:
        return -result[0]['score']

# Load the news data
news_df = pd.read_csv(r"Indian-FINBERT-LSTM/indian_finance_news_past_month.csv")

# Assuming the first column is 'Date' and the rest are news headlines
date_column = news_df.columns[0]
headline_columns = news_df.columns[1:]

# Create a new dataframe with the desired structure
result_df = pd.DataFrame(columns=['Date'] + [f'News {i+1}' for i in range(10)] + ['FinBERT score'])

for _, row in news_df.iterrows():
    date = row[date_column]
    headlines = row[headline_columns].tolist()
    score = FinBERT_sentiment_score(headlines)
    
    # Ensure we have exactly 10 headlines, padding with empty strings if necessary
    headlines = headlines[:10] + [''] * (10 - len(headlines))
    
    new_row = [date] + headlines + [score]
    result_df.loc[len(result_df)] = new_row

# Save the results
result_df.to_csv("indian_sentiment.csv", index=False)
print("Sentiment analysis complete. Results saved to indian_sentiment.csv")