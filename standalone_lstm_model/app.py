import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
import matplotlib
import logging

matplotlib.use('Agg') # Use Agg backend for non-interactive plotting

# --- Add Transformers and Torch ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# ---------------------------------

# --- Add Pandas Option for Future Behavior ---
pd.set_option('future.no_silent_downcasting', True)
# -------------------------------------------

app = Flask(__name__)
app.logger.setLevel(logging.INFO) # Now 'logging' is defined

# --- Configuration ---
# Use environment variable first, fallback to the provided key if not set
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")
if NEWS_API_KEY == "YOUR_NEWSAPI_KEY" and "NEWS_API_KEY" not in os.environ:
    app.logger.warning("Using hardcoded NEWS_API_KEY. Consider setting it as an environment variable for security.")
MODEL_PATH = 'best_finbert_lstm_model.h5'
# SENTIMENT_CSV_PATH = "finbert_sentiment_results.csv" # No longer needed

# --- Load FinBERT Model Globally (more efficient) ---
try:
    finbert_model_name = "ProsusAI/finbert"
    tokenizer_finbert = AutoTokenizer.from_pretrained(finbert_model_name)
    model_finbert = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
    app.logger.info("FinBERT model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading FinBERT model: {e}", exc_info=True)
    # Depending on your needs, you might want to exit or disable sentiment analysis here
    tokenizer_finbert = None
    model_finbert = None
# ----------------------------------------------------

# --- Helper Functions ---

def generate_signal(current_price, predicted_price, threshold_buy=1.0, threshold_sell=1.0):
    """Generates Buy/Sell/Hold signal based on price change percentage."""
    if current_price == 0: # Avoid division by zero
        return "HOLD", 0.0, "hold"
    percent_change = (predicted_price - current_price) / current_price * 100

    if percent_change >= threshold_buy:
        return "BUY", percent_change, "buy"
    elif percent_change <= -threshold_sell:
        return "SELL", percent_change, "sell"
    else:
        return "HOLD", percent_change, "hold"


def create_plot(dates, actual, predicted):
    """Generates a plot and returns base64 encoded string."""
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label='Actual Price', color='blue', marker='.')
    plt.plot(dates, predicted, label='Predicted Price', color='green', linestyle='--', marker='.')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free memory
    return plot_url

# --- Sentiment Analysis Functions ---

def analyze_sentiment_finbert(text):
    """Analyzes a single text string using the loaded FinBERT model."""
    if not model_finbert or not tokenizer_finbert or not text or not isinstance(text, str):
        # Return neutral if model isn't loaded or text is invalid
        return {'label': 'neutral', 'score': 0.0, 'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}}

    try:
        inputs = tokenizer_finbert(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model_finbert(**inputs)
        scores = torch.softmax(outputs.logits, dim=1).squeeze()
        labels = ['positive', 'negative', 'neutral']
        sentiment_idx = torch.argmax(scores).item()
        return {
            'label': labels[sentiment_idx],
            'score': scores[sentiment_idx].item(),
            'scores': {label: score.item() for label, score in zip(labels, scores)}
        }
    except Exception as e:
        app.logger.error(f"Error during FinBERT analysis for text '{text[:50]}...': {e}")
        return {'label': 'neutral', 'score': 0.0, 'scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}}


def perform_sentiment_analysis_on_articles(articles):
    """Processes a list of news articles and returns aggregated daily sentiment."""
    sentiments = []
    app.logger.info(f"Performing sentiment analysis on {len(articles)} articles.")
    count = 0
    for article in articles:
        # ... (extract title, description, published_at) ...
        title = article.get('title', '')
        description = article.get('description', '')
        published_at = article.get('publishedAt')
        text_to_analyze = title if title else ""
        if description:
            text_to_analyze += ". " + description
        text_to_analyze = text_to_analyze[:512]

        if text_to_analyze and published_at:
            try:
                # *** Convert to datetime and make timezone-aware (UTC) here ***
                published_dt = pd.to_datetime(published_at) # This might already be UTC depending on NewsAPI format
                if published_dt.tzinfo is None:
                     published_dt = published_dt.tz_localize('UTC') # Assume UTC if naive
                else:
                     published_dt = published_dt.tz_convert('UTC') # Convert to UTC if it has another timezone

                sentiment_result = analyze_sentiment_finbert(text_to_analyze)
                sentiments.append({
                    # Store the UTC datetime object
                    'datetime_utc': published_dt,
                    'positive': sentiment_result['scores'].get('positive', 0.0),
                    'negative': sentiment_result['scores'].get('negative', 0.0),
                    'neutral': sentiment_result['scores'].get('neutral', 0.0),
                })
                count += 1
            except Exception as e:
                 app.logger.error(f"Error processing article published at {published_at}: {e}")

    app.logger.info(f"Successfully analyzed sentiment for {count} articles.")
    if not sentiments:
        return pd.DataFrame(columns=['date', 'daily_positive', 'daily_negative', 'daily_neutral'])

    sentiment_df = pd.DataFrame(sentiments)

    # --- Aggregate daily scores ---
    # Group by the date part of the UTC datetime
    daily_sentiment = sentiment_df.groupby(sentiment_df['datetime_utc'].dt.date).agg(
        daily_positive=('positive', 'mean'),
        daily_negative=('negative', 'mean'),
        daily_neutral=('neutral', 'mean')
    ).reset_index()

    # Rename the grouping column back to 'date' and ensure it's datetime
    daily_sentiment.rename(columns={'datetime_utc': 'date'}, inplace=True)
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date']) # Convert date back to datetime64[ns] (naive)

    app.logger.info(f"Aggregated sentiment into {len(daily_sentiment)} daily records. Date column type: {daily_sentiment['date'].dtype}")
    return daily_sentiment

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main input form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    """
    Fetches data, performs real-time sentiment analysis, runs prediction,
    generates signal, and renders the result page.
    """
    if not model_finbert:
         return "FinBERT model not loaded. Cannot perform sentiment analysis.", 500

    try:
        ticker = request.form.get('ticker')
        query = request.form.get('query')
        from_date_str = request.form.get('fromDate')
        to_date_str = request.form.get('toDate')

        app.logger.info(f"Received analysis request for Ticker: {ticker}, Query: {query}, From: {from_date_str}, To: {to_date_str}")

        if not ticker or not from_date_str or not to_date_str:
            return "Missing required form fields (ticker, fromDate, toDate).", 400

        news_query = query if query else ticker # Use ticker for news query if not provided

        # --- 1. Fetch Stock Data using yfinance ---
        app.logger.info(f"Fetching stock data for {ticker} from {from_date_str} to {to_date_str}")
        # Explicitly set auto_adjust=False to get 'Adj Close' which might be more consistent
        # Or keep auto_adjust=True (default) and handle the suffixed columns
        stock_data = yf.download(ticker, start=from_date_str, end=to_date_str, auto_adjust=False) # Try auto_adjust=False

        # --- Column Flattening (Keep simple version) ---
        if isinstance(stock_data.columns, pd.MultiIndex):
            app.logger.info("Detected MultiIndex columns from yfinance. Flattening.")
            stock_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in stock_data.columns.values]
        # --- End Column Flattening ---

        if stock_data.empty:
            app.logger.warning(f"yfinance returned empty data for {ticker} in range {from_date_str}-{to_date_str}")
            return f"Could not fetch stock data for ticker '{ticker}'. Please check the ticker and date range.", 404

        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.normalize()
        app.logger.info(f"Processed stock data columns: {stock_data.columns.tolist()}")

        # --- *** Identify the correct Close column name *** ---
        # Prioritize 'Adj Close' if auto_adjust=False, then 'Close', then suffixed versions
        close_col_name = None
        possible_names = ['Adj Close', 'Close'] # Add other potential names if needed
        # Add suffixed names based on the ticker
        possible_names.extend([f'Adj Close_{ticker}', f'Close_{ticker}'])
        # Add names from the actual columns in case of unexpected formatting
        possible_names.extend(stock_data.columns.tolist())

        for name in possible_names:
             if name in stock_data.columns:
                  close_col_name = name
                  app.logger.info(f"Identified Close column as: '{close_col_name}'")
                  break # Found the column

        if not close_col_name:
             app.logger.error(f"Could not identify a Close or Adj Close column in stock data. Columns: {stock_data.columns.tolist()}")
             return "Internal error: Failed to identify price column.", 500
        # --- *** End Identify Close column name *** ---


        # --- 2. Fetch News Data ---
        app.logger.info(f"Fetching news for query '{news_query}' from {from_date_str} to {to_date_str}")
        # Limit page size to avoid excessive processing time, adjust as needed
        url = f'https://newsapi.org/v2/everything?q={news_query}&from={from_date_str}&to={to_date_str}&apiKey={NEWS_API_KEY}&language=en&sortBy=relevancy&pageSize=50'
        news_articles = []
        try:
            response = requests.get(url, timeout=10) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            news_data = response.json()
            if news_data.get('status') == 'ok':
                news_articles = news_data.get('articles', [])
                app.logger.info(f"Fetched {len(news_articles)} news articles (total results: {news_data.get('totalResults')}).")
            else:
                 app.logger.error(f"NewsAPI error: {news_data.get('message')}")
                 # Optionally return an error or proceed without sentiment
                 # return f"NewsAPI error: {news_data.get('message')}", 500
        except requests.exceptions.RequestException as e:
            app.logger.error(f"NewsAPI request failed: {e}")
            # Optionally return an error or proceed without sentiment
            # return f"Failed to fetch news: {e}", 500

        # --- 3. Perform Real-time Sentiment Analysis ---
        if news_articles:
            daily_sentiment_df = perform_sentiment_analysis_on_articles(news_articles)
        else:
            app.logger.warning("No news articles found, proceeding without sentiment data.")
            # Create an empty df with correct columns if no news
            daily_sentiment_df = pd.DataFrame(columns=['date', 'daily_positive', 'daily_negative', 'daily_neutral'])


        # --- 4. Merge Stock and Sentiment Data ---
        app.logger.info("Merging stock data and daily sentiment data.")
        app.logger.info(f"Stock data columns before merge: {stock_data.columns.tolist()}")
        app.logger.info(f"Sentiment data columns before merge: {daily_sentiment_df.columns.tolist()}")

        merged_data = pd.merge(stock_data, daily_sentiment_df, left_on='Date', right_on='date', how='left')

        # Log columns *after* merge
        app.logger.info(f"Merged data columns: {merged_data.columns.tolist()}")

        # Fill missing sentiment
        merged_data['daily_positive'] = merged_data['daily_positive'].fillna(0.0)
        merged_data['daily_negative'] = merged_data['daily_negative'].fillna(0.0)
        merged_data['daily_neutral'] = merged_data['daily_neutral'].fillna(1.0) # Fill with neutral
        merged_data.drop(columns=['date'], inplace=True, errors='ignore')
        app.logger.info(f"Merged data columns: {merged_data.columns.tolist()}")

        if merged_data.empty:
            app.logger.error("Merged data is empty after joining stock and sentiment.")
            return "No data available after merging stock prices and sentiment.", 404
        app.logger.info(f"Merged data contains {len(merged_data)} rows.")


        # --- 5. Prepare Data for LSTM Model ---
        required_sentiment_cols = ['daily_positive', 'daily_negative', 'daily_neutral']
        # Check for the identified close column name *after* merge and fillna
        if close_col_name not in merged_data.columns:
             app.logger.error(f"CRITICAL: Identified Close column '{close_col_name}' is missing after merge/fillna. Columns: {merged_data.columns.tolist()}")
             return f"Internal error: Processing failed to preserve price column '{close_col_name}'.", 500

        features = merged_data[required_sentiment_cols].values
        # *** Use the identified close_col_name ***
        target = merged_data[close_col_name].values.reshape(-1, 1)

        # Scale features and target
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler_X.fit_transform(features)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_y.fit(target) # Fit scaler on the actual target data

        # Reshape for LSTM
        # Assuming lookback=1 for simplicity here, adjust if your model expects more timesteps
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        app.logger.info(f"Prepared data for LSTM with shape: {X_reshaped.shape}")


        # --- 6. Load LSTM Model and Predict ---
        app.logger.info(f"Loading LSTM model from {MODEL_PATH}")
        try:
            model_lstm = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            app.logger.error(f"Error loading LSTM model from '{MODEL_PATH}': {e}", exc_info=True)
            return f"Error loading prediction model: {str(e)}", 500

        app.logger.info("Making predictions with LSTM model.")
        predictions_scaled = model_lstm.predict(X_reshaped)
        predicted_prices = scaler_y.inverse_transform(predictions_scaled) # Use the target scaler


        # --- 7. Generate Signal for the Latest Data Point ---
        if len(merged_data) == 0 or len(predicted_prices) == 0:
             app.logger.warning("Not enough data points after processing to generate a signal.")
             return "Not enough data to generate a signal after processing.", 500

        # *** Use the identified close_col_name ***
        latest_actual_price = merged_data[close_col_name].iloc[-1]
        latest_predicted_price = predicted_prices[-1][0]

        signal, percent_change, signal_class = generate_signal(latest_actual_price, latest_predicted_price)
        app.logger.info(f"Generated signal: {signal} ({percent_change:.2f}%)")


        # --- 8. Generate Plot ---
        plot_limit = min(30, len(merged_data))
        app.logger.info(f"Generating plot for the last {plot_limit} data points using column '{close_col_name}'.")
        plot_url = create_plot(
            merged_data['Date'].iloc[-plot_limit:],
            # *** Use the identified close_col_name ***
            merged_data[close_col_name].iloc[-plot_limit:],
            predicted_prices[-plot_limit:].flatten()
        )

        # --- 9. Render Result Page ---
        app.logger.info("Rendering signal page.")
        return render_template(
            'signal.html',
            ticker=ticker.upper(),
            signal=signal,
            signal_class=signal_class,
            current_price=latest_actual_price,
            predicted_price=latest_predicted_price,
            percent_change=percent_change,
            plot_url=plot_url
        )

    except FileNotFoundError as e:
         app.logger.error(f"File not found error: {e}", exc_info=True)
         return f"A required data file was not found: {e}", 500
    except ValueError as e:
         app.logger.error(f"Value error during analysis: {e}", exc_info=True)
         return f"An error occurred during data processing: {e}", 400
    except Exception as e:
        app.logger.error(f"Unexpected error during /analyze: {e}", exc_info=True)
        return f"An unexpected error occurred. Please check logs.", 500

# --- Add Logging Configuration ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --------------------------------

if __name__ == '__main__':
    # Consider using a production server like gunicorn instead of Flask's built-in server
    app.run(debug=False, host='0.0.0.0') # Run on all interfaces, disable debug for production