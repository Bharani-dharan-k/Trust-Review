import requests
import json
import pandas as pd
import os

def download_yelp_dataset():
    """Download a subset of Yelp reviews (mocked here due to API restrictions)"""
    yelp_url = "https://www.yelp.com/dataset/download"  # Requires manual download or API
    # For simplicity, simulate a subset since direct download requires registration
    if not os.path.exists('datasets/yelp_reviews.csv'):
        yelp_data = {
            'text': [
                "Amazing food, great service, will come back!",
                "Terrible place, food was cold, avoid!",
                "Perfect dining experience, highly recommend.",
                "Okay food, nothing special, overpriced."
            ],
            'stars': [5, 1, 5, 3],
            'useful': [5, 2, 0, 3]  # Low 'useful' votes might indicate fake
        }
        df = pd.DataFrame(yelp_data)
        # Infer labels: Low useful votes + extreme rating = potential fake
        df['label'] = df.apply(lambda row: 0 if (row['stars'] in [1, 5] and row['useful'] < 2) else 1, axis=1)
        os.makedirs('datasets', exist_ok=True)
        df.to_csv('datasets/yelp_reviews.csv', index=False)
        st.info("Created mock Yelp reviews dataset (replace with real subset)")

def download_amazon_dataset():
    """Download a subset of Amazon reviews (mocked here)"""
    amazon_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
    # Simulate a subset
    if not os.path.exists('datasets/amazon_reviews.csv'):
        amazon_data = {
            'text': [
                "Best headphones ever, super fast delivery!",
                "Works fine, good for price, not amazing.",
                "Awful product, broke in a day, waste of money.",
                "Great sound quality, very happy with purchase."
            ],
            'stars': [5, 3, 1, 4],
            'verified': [False, True, True, True]  # Verified purchase as proxy for genuine
        }
        df = pd.DataFrame(amazon_data)
        df['label'] = df['verified'].apply(lambda x: 1 if x else 0)
        df.drop(columns=['verified'], inplace=True)
        os.makedirs('datasets', exist_ok=True)
        df.to_csv('datasets/amazon_reviews.csv', index=False)
        st.info("Created mock Amazon reviews dataset (replace with real subset)")