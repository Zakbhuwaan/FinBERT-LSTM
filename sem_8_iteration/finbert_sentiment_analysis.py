import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax

# Load FinBERT model and tokenizer
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to get sentiment scores using FinBERT
def get_sentiment_score(text):
    if pd.isna(text) or text.strip() == "":
        return pd.Series([None, None, None])  # Return NaNs for missing text

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits.detach().cpu().numpy()[0]
    probabilities = softmax(logits)  # Convert logits to probabilities

    return pd.Series(probabilities)  # Return [positive, negative, neutral] scores

# Load dataset
df = pd.read_csv(r"sem_8_iteration\nifty_50_stock_news.csv")

# Rename columns if necessary
df.columns = ["date"] + [f"news {i}" for i in range(1, 11)]  # Ensure correct naming

# Apply sentiment analysis on each news column
for col in [f"news {i}" for i in range(1, 11)]:
    df[[f"{col}_positive", f"{col}_negative", f"{col}_neutral"]] = df[col].apply(get_sentiment_score)

# Compute daily average sentiment scores
df["daily_positive"] = df[[f"news {i}_positive" for i in range(1, 11)]].mean(axis=1)
df["daily_negative"] = df[[f"news {i}_negative" for i in range(1, 11)]].mean(axis=1)
df["daily_neutral"] = df[[f"news {i}_neutral" for i in range(1, 11)]].mean(axis=1)

# Save results to CSV
df.to_csv("finbert_sentiment_results.csv", index=False)

print("Sentiment analysis completed! Results saved in 'finbert_sentiment_results.csv'.")
