import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import xgboost as xgb
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import re
import time
import random
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Set Streamlit page config as the FIRST command
st.set_page_config(page_title="TrustReview", layout="wide")

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
sid = SentimentIntensityAnalyzer()

# Setup requests session with retry logic
session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Function to convert product URL to review URL
def get_review_url(url):
    if "product-reviews" in url:
        return url
    # Amazon URL
    asin_match = re.search(r'/dp/([A-Z0-9]{10})', url)
    if asin_match:
        asin = asin_match.group(1)
        return f"https://www.amazon.in/product-reviews/{asin}"
    # Flipkart URL
    item_match = re.search(r'/p/(itm[a-z0-9]+)', url)
    pid_match = re.search(r'pid=([A-Z0-9]+)', url)
    if item_match and pid_match:
        item_id = item_match.group(1)
        pid = pid_match.group(1)
        return f"https://www.flipkart.com/product-reviews/{item_id}?pid={pid}&marketplace=FLIPKART"
    return url

# Function to scrape reviews (with Selenium fallback and pagination)
def scrape_reviews(url):
    reviews = []
    page = 1
    review_url = get_review_url(url)
    progress_bar = st.progress(0)

    while True:
        page_url = review_url if page == 1 else review_url + f"&page={page}"
        st.write(f"Scraping reviews from {page_url}...")

        # Try requests first
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/129.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.flipkart.com/"
        }
        try:
            response = session.get(page_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            review_elements = soup.select("div.t-zTKy div div, div.ZmyHeo, div.qzwNOZ")
            st.write(f"Page {page} (requests): Found {len(review_elements)} review elements")
        except:
            review_elements = []

        # Fallback to Selenium if no reviews
        if not review_elements:
            st.write(f"Switching to browser-based scraping for page {page}...")
            try:
                options = Options()
                options.add_argument("--headless")
                driver = webdriver.Chrome(options=options)
                driver.get(page_url)
                time.sleep(5)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                review_elements = soup.select("div.t-zTKy div div, div.ZmyHeo, div.qzwNOZ")
                st.write(f"Page {page} (Selenium): Found {len(review_elements)} review elements")
                driver.quit()
            except Exception as e:
                st.warning(f"Selenium failed for page {page}: {e}")
                break

        if not review_elements:
            st.warning(f"No more reviews found on page {page}.")
            break

        for review in review_elements:
            try:
                text = review.text.strip()
                parent = review.find_parent("div", class_="col EPCmJX")
                rating_elem = parent.find("div", class_="XQDdHH") if parent else None
                rating_val = float(rating_elem.text.strip()) if rating_elem and rating_elem.text.strip() else 0
                date_elem = parent.find("div", class_="R3kR3O") if parent else None
                date = date_elem.text.strip() if date_elem else ""
                author_elem = parent.find("p", class_="z9E0FP") if parent else None
                author = author_elem.text.strip() if author_elem else "Unknown"

                if text and len(text) > 10:
                    reviews.append({
                        "text": text,
                        "rating": rating_val,
                        "date": date,
                        "author": author
                    })
            except:
                continue

        progress_bar.progress(min(page / 50, 1.0))
        st.write(f"Collected {len(reviews)} reviews so far...")
        page += 1
        time.sleep(random.uniform(5, 10))

    if reviews:
        st.success(f"Collected {len(reviews)} reviews!")
    else:
        st.error("No reviews found.")
    return reviews

# Function to extract features
def extract_features(reviews):
    if not reviews:
        return None, None

    df = pd.DataFrame(reviews)
    features = []
    fake_phrases = ["great product", "highly recommend", "best ever", "awesome", "terrible", "worst"]
    positive_words = ["good", "great", "awesome", "excellent", "fantastic"]
    negative_words = ["bad", "poor", "terrible", "awful", "horrible"]

    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    texts = df['text'].tolist() + fake_phrases
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf_matrix[-len(fake_phrases):], tfidf_matrix[:-len(fake_phrases)])
        fake_sim_scores = similarities.max(axis=0)
    except ValueError:
        fake_sim_scores = np.zeros(len(df))

    rating_variance = df['rating'].var() if len(df['rating']) > 1 else 0
    try:
        dates = pd.to_datetime(df['date'], errors='coerce')
        date_range = (dates.max() - dates.min()).days
        review_frequency = len(dates) / max(date_range, 1)
        date_counts = dates.dt.date.value_counts()
        is_burst = int(date_counts.max() > len(dates) * 0.3) if not date_counts.empty else 0
    except:
        review_frequency = 0
        is_burst = 0

    author_counts = df['author'].value_counts().to_dict()

    for i, row in df.iterrows():
        text = row['text']
        blob = TextBlob(text)
        vader_scores = sid.polarity_scores(text)
        tokens = nltk.word_tokenize(text.lower())
        sentences = nltk.sent_tokenize(text)

        text_length = len(text)
        positive_count = sum(1 for word in tokens if word in positive_words)
        negative_count = sum(1 for word in tokens if word in negative_words)
        detail_score = len(re.findall(r'\b(food|service|staff|price|quality)\b', text.lower()))
        grammar_complexity = len(sentences) / max(len(tokens), 1)
        repetition_score = len(tokens) / len(set(tokens)) if tokens else 1.0
        specificity = len(set(re.findall(r'\b(pasta|flavors|staff|ambiance|penny)\b', text.lower())))
        sentiment_star_diff = abs(vader_scores['compound'] - (row['rating'] - 1) / 2.5)
        is_extreme = int(row['rating'] in [1, 5])
        emotional_intensity = positive_count + negative_count

        feature = {
            'text_length': text_length,
            'fake_pattern_sim': fake_sim_scores[i] if i < len(fake_sim_scores) else 0,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'detail_score': detail_score,
            'grammar_complexity': grammar_complexity,
            'repetition_score': repetition_score,
            'specificity': specificity,
            'blob_polarity': blob.sentiment.polarity,
            'blob_subjectivity': blob.sentiment.subjectivity,
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'vader_pos': vader_scores['pos'],
            'vader_compound': vader_scores['compound'],
            'stars': row['rating'],
            'num_reviews': author_counts.get(row['author'], 1),
            'review_frequency': review_frequency,
            'rating_variance': rating_variance,
            'is_burst': is_burst,
            'sentiment_star_diff': sentiment_star_diff,
            'is_extreme': is_extreme,
            'emotional_intensity': emotional_intensity
        }
        features.append(feature)

    return df, pd.DataFrame(features)

# Function to predict fake reviews
def predict_reviews(features_df):
    if features_df.empty:
        return None
    try:
        with open('models/xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.write("Model loaded successfully from models/xgb_model.pkl")  # Debug message
        expected_features = [
            'text_length', 'fake_pattern_sim', 'positive_count', 'negative_count', 'detail_score',
            'grammar_complexity', 'repetition_score', 'specificity', 'blob_polarity', 'blob_subjectivity',
            'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'stars', 'num_reviews',
            'review_frequency', 'rating_variance', 'is_burst', 'sentiment_star_diff', 'is_extreme',
            'emotional_intensity'
        ]
        features_df = features_df[expected_features]
        predictions = model.predict(features_df)
        return predictions  # 0 = genuine, 1 = fake
    except FileNotFoundError:
        st.error("XGBoost model file not found in 'models' folder. Please ensure 'xgb_model.pkl' is available.")
        return None
    except Exception as e:
        st.error(f"Error predicting reviews: {e}")
        return None

# Streamlit App
st.title("TrustReview")
st.markdown("Analyze product reviews from Amazon or Flipkart to detect fake reviews using AI.")

url = st.text_input("Enter Product or Review Page URL", 
                    placeholder="e.g., https://www.amazon.in/product-reviews/B08L5T31M7 or https://www.flipkart.com/product-reviews/itma94644ecf11b8?pid=BLCG7GYAGHZ5HMQY")

if st.button("Analyze Reviews"):
    if url:
        with st.spinner("Fetching and analyzing reviews..."):
            reviews = scrape_reviews(url)
            if reviews:
                df, features_df = extract_features(reviews)
                if df is not None and features_df is not None:
                    predictions = predict_reviews(features_df)
                    if predictions is not None:
                        df['label'] = ['fake' if pred == 1 else 'real' for pred in predictions]

                        # Save dataset
                        os.makedirs("datasets", exist_ok=True)
                        dataset_path = "datasets/reviews.csv"
                        df.to_csv(dataset_path, index=False)
                        st.success(f"Saved {len(df)} reviews to {dataset_path}")

                        # Summary
                        st.subheader("Review Analysis Results")
                        summary = df['label'].value_counts()
                        total = len(df)
                        st.write(f"*Total Reviews*: {total}")
                        for label in ['real', 'fake']:
                            count = summary.get(label, 0)
                            st.write(f"{label.capitalize()}: {count} ({count/total*100:.2f}%)")

                        # Pie Chart
                        fig, ax = plt.subplots()
                        labels = [label.capitalize() for label in summary.index]
                        ax.pie(summary.values, labels=labels, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')
                        st.pyplot(fig)

                        # Fake Review Samples
                        st.subheader("Fake Review Samples")
                        fake_reviews = df[df['label'] == 'fake']
                        if not fake_reviews.empty:
                            st.write("The following reviews are predicted as fake:")
                            for _, row in fake_reviews.head(5).iterrows():
                                st.markdown(f"*Text: {row['text']}*")
                        else:
                            st.info("No fake reviews detected.")

                        # Download Button
                        st.subheader("Download Dataset")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Reviews CSV",
                            data=csv,
                            file_name="reviews_dataset.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Failed to predict reviews. Check model file.")
                else:
                    st.error("Failed to process reviews.")
            else:
                st.error("No reviews found. Ensure the URL is correct and the product has reviews.")
    else:
        st.error("Please enter a valid URL.")