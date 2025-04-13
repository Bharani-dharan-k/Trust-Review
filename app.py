import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import xgboost as xgb
import joblib
import re
import os
from collections import defaultdict
from datetime import datetime, timedelta
import random
import nltk
import io

nltk.download('punkt', quiet=True)

# Set up page config
st.set_page_config(
    page_title="Review Authenticator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Translator
translator = Translator()

# Language selection
language_options = {name: code for code, name in LANGUAGES.items()}
default_lang = 'en'

# Sidebar for language selection
st.sidebar.title("Settings")
selected_language = st.sidebar.selectbox("Select Language", options=list(language_options.keys()), index=list(language_options.keys()).index('english'))
target_lang_code = language_options[selected_language]

# Function to translate text with error handling
def translate_text(text, dest_lang='en'):
    if not text or dest_lang == 'en':
        return text
    try:
        translated = translator.translate(text, dest=dest_lang)
        return translated.text if translated else text
    except Exception as e:
        st.warning(f"Translation failed for '{text}': {e}. Using original text.")
        return text

# Cache translations to avoid repeated API calls
@st.cache_data
def cache_translations(ui_strings, dest_lang):
    return {key: translate_text(value, dest_lang) for key, value in ui_strings.items()}

# UI strings
ui_strings = {
    'title': "Review Authenticator",
    'description': "Detect fake reviews with AI-powered analysis",
    'enter_review': "Enter Review Details",
    'product_review': "Product Review",
    'star_rating': "Star Rating",
    'analyze_btn': "Analyze Review",
    'analysis_complete': "Analysis Complete!",
    'prediction_genuine': "Genuine Review ✅",
    'prediction_fake': "Fake Review 🚩",
    'confidence': "Confidence",
    'sentiment': "Sentiment",
    'sentiment_score': "Score",
    'user_rating': "User Rating",
    'matches_sentiment': "👍 Matches sentiment",
    'mismatch': "⚠ Mismatch",
    'language_tab': "Language",
    'sentiment_tab': "Sentiment",
    'history_tab': "History",
    'features_tab': "Features",
    'language_analysis': "Language Analysis",
    'detected_language': "Detected Language",
    'translated': "Translated",
    'not_needed': "Not needed",
    'sentiment_vs_rating': "Sentiment vs Rating",
    'reviewer_history': "Reviewer History",
    'no_history': "No history available",
    'feature_breakdown': "Feature Breakdown",
    'please_enter_review': "Please enter a review",
    'voice_output': "Play Voice Output"
}

# Translate UI strings with caching
translated_ui = cache_translations(ui_strings, target_lang_code)

# Download and save datasets
def download_datasets():
    os.makedirs('datasets', exist_ok=True)
    
    if not os.path.exists('datasets/indic_sentiment.csv'):
        mock_indic = pd.DataFrame({
            'text': ["मस्त प्रोडक्ट, बहुत अच्छा है।", "खराब क्वालिटी, बेकार है।", "The product is great, love it!", "Bad experience, won't buy again."],
            'label': ['positive', 'negative', 'positive', 'negative'],
            'language': ['hi', 'hi', 'en', 'en']
        })
        mock_indic.to_csv('datasets/indic_sentiment.csv', index=False)
        st.info("Created mock IndicSentiment dataset")

    if not os.path.exists('datasets/amazon_polarity.csv'):
        mock_amazon = pd.DataFrame({
            'text': ["Really happy with this purchase, works well.", "Terrible product, broke after one use.", "Decent quality, worth the price.", "Not satisfied, poor performance."],
            'label': [1, 0, 1, 0]
        })
        mock_amazon.to_csv('datasets/amazon_polarity.csv', index=False)
        st.info("Created mock amazon_polarity dataset")

# Load datasets with column name standardization
@st.cache_data
def load_datasets():
    datasets = {}
    possible_text_cols = ['text', 'review_text', 'content']
    for file in ['indic_sentiment.csv', 'amazon_polarity.csv', 'yelp_dataset.csv']:
        if os.path.exists(f'datasets/{file}'):
            df = pd.read_csv(f'datasets/{file}')
            text_col = next((col for col in possible_text_cols if col in df.columns), None)
            if text_col:
                if text_col != 'text':
                    df = df.rename(columns={text_col: 'text'})
                datasets[file.split('.')[0]] = df
            else:
                st.error(f"No review text column found in {file}. Expected one of {possible_text_cols}")
    return datasets

# Fake Review Detector
class FakeReviewDetector:
    def __init__(self):
        self.feature_cols = [
            'text_length', 'fake_pattern_sim', 'positive_count', 'negative_count', 'detail_score',
            'grammar_complexity', 'repetition_score', 'specificity', 'blob_polarity', 'blob_subjectivity',
            'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'stars', 'num_reviews',
            'review_frequency', 'rating_variance', 'is_burst', 'sentiment_star_diff', 'is_extreme',
            'emotional_intensity'
        ]
        self.setup_nlp_pipelines()
        self.initialize_mock_data()
        self.initialize_tfidf()
        self.load_models()

    def initialize_tfidf(self):
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        sample_texts = self.fake_patterns + self.genuine_patterns
        self.tfidf.fit(sample_texts)
        self.fake_vectors = self.tfidf.transform(self.fake_patterns)
        self.genuine_vectors = self.tfidf.transform(self.genuine_patterns)

    def load_models(self):
        model_path = 'models/xgb_model.pkl'
        if os.path.exists(model_path):
            self.xgb_model = joblib.load(model_path)
            expected_features = len(self.feature_cols)
            model_features = self.xgb_model.get_booster().num_features()
            if model_features != expected_features:
                st.warning(f"Feature mismatch: Model expects {model_features}, code expects {expected_features}. Retraining.")
                self.train_model()
        else:
            self.train_model()

    def setup_nlp_pipelines(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.translator = Translator()
        self.positive_words = {
            'hindi': ['अच्छा', 'मस्त', 'शानदार', 'बेहतरीन', 'उत्कृष्ट'],
            'english': ['awesome', 'fantastic', 'excellent', 'great', 'love', 'amazing', 'wonderful']
        }
        self.negative_words = {
            'hindi': ['खराब', 'भयानक', 'बेकार', 'घटिया', 'निराशाजनक'],
            'english': ['bad', 'worst', 'horrible', 'terrible', 'hate', 'awful']
        }

    def initialize_mock_data(self):
        self.user_history = defaultdict(list)
        self.fake_patterns = ["best ever must buy", "perfect highly recommend", "excellent fast delivery"]
        self.genuine_patterns = ["good but could be better", "decent for price", "works as expected"]

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return 'en'

    def translate_to_english(self, text, src_lang):
        if src_lang == 'en':
            return text
        try:
            translated = self.translator.translate(text, src=src_lang, dest='en')
            return translated.text if translated else text
        except:
            return text

    def extract_features(self, text, stars, user_id="default"):
        lang = self.detect_language(text)
        translated_text = self.translate_to_english(text, lang)
        blob = TextBlob(translated_text)
        vader_scores = self.sentiment_analyzer.polarity_scores(translated_text)
        translated_vector = self.tfidf.transform([translated_text])

        text_length = len(text)
        fake_sim = max(cosine_similarity(translated_vector, self.fake_vectors)[0]) if self.fake_vectors.shape[0] > 0 else 0
        positive_count = sum(1 for word in re.findall(r'\w+', translated_text.lower()) 
                            for lang_words in self.positive_words.values() if word in lang_words)
        negative_count = sum(1 for word in re.findall(r'\w+', translated_text.lower()) 
                            for lang_words in self.negative_words.values() if word in lang_words)
        detail_score = len(re.findall(r'\b(food|pasta|service|staff|ambiance|price|quality)\b', translated_text.lower()))
        history = self.get_user_history(user_id)
        num_reviews = len(history)
        review_frequency = len([r for r in history if (datetime.now() - r['date']).days <= 7])
        rating_variance = np.var([r['stars'] for r in history]) if history else 1.0
        is_burst = review_frequency > 3
        sentiment_star_diff = abs(vader_scores['compound'] - (stars-1)/2.5)
        is_extreme = 1 if stars in [1, 5] else 0
        emotional_intensity = positive_count + negative_count

        grammar_complexity = len(blob.sentences) / (len(blob.words) + 1)
        words = re.findall(r'\w+', translated_text.lower())
        repetition_score = len(words) / len(set(words)) if words else 1.0
        specificity = len(set(re.findall(r'\b(pasta|flavors|staff|ambiance|penny)\b', translated_text.lower())))

        features = {
            'text_length': text_length,
            'fake_pattern_sim': fake_sim,
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
            'stars': stars,
            'num_reviews': num_reviews,
            'review_frequency': review_frequency,
            'rating_variance': rating_variance,
            'is_burst': int(is_burst),
            'sentiment_star_diff': sentiment_star_diff,
            'is_extreme': is_extreme,
            'emotional_intensity': emotional_intensity
        }
        return features, translated_text

    def get_user_history(self, user_id="default"):
        if user_id not in self.user_history:
            num_reviews = random.randint(3, 10)
            for _ in range(num_reviews):
                days_ago = random.randint(1, 90)
                date = datetime.now() - timedelta(days=days_ago)
                stars = random.randint(1, 5)
                is_fake = random.random() > 0.7
                text = random.choice(self.fake_patterns) if is_fake else random.choice(self.genuine_patterns)
                self.user_history[user_id].append({
                    'date': date,
                    'text': text,
                    'stars': stars,
                    'is_fake': is_fake,
                    'sentiment': self.sentiment_analyzer.polarity_scores(text)['compound']
                })
        return self.user_history[user_id]

    def train_model(self):
        datasets = load_datasets()
        features_list = []
        labels = []

        if 'indic_sentiment' in datasets:
            df = datasets['indic_sentiment']
            for _, row in df.iterrows():
                text = row['text']
                label = 1 if row['label'] == 'positive' else 0
                stars = 5 if row['label'] == 'positive' else 1
                features, _ = self.extract_features(text, stars)
                features_list.append(features)
                labels.append(label)

        if 'amazon_polarity' in datasets:
            df = datasets['amazon_polarity']
            for _, row in df.iterrows():
                text = row['text']
                label = row['label']
                stars = 5 if label == 1 else 1
                features, _ = self.extract_features(text, stars)
                features_list.append(features)
                labels.append(label)

        if 'yelp_dataset' in datasets:
            df = datasets['yelp_dataset']
            if 'label' not in df.columns:
                df['label'] = df.apply(lambda row: 0 if (len(set(re.findall(r'\w+', row['text'].lower()))) < 10 and 
                                                        len(re.findall(r'\b(food|service|staff|price)\b', row['text'].lower())) < 2) 
                                                      else 1, axis=1)
            for _, row in df.iterrows():
                text = row['text']
                stars = row['stars']
                label = row['label']
                features, _ = self.extract_features(text, stars)
                features_list.append(features)
                labels.append(label)

        if not features_list:
            st.error("No data available. Using dummy data.")
            dummy_data = {
                'text_length': [10, 15, 8, 20, 12, 18],
                'fake_pattern_sim': [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
                'positive_count': [3, 0, 4, 0, 1, 0],
                'negative_count': [0, 4, 0, 5, 1, 3],
                'detail_score': [2, 0, 3, 0, 1, 0],
                'grammar_complexity': [0.2, 0.1, 0.3, 0.15, 0.25, 0.2],
                'repetition_score': [1.5, 1.2, 1.8, 1.1, 1.4, 1.3],
                'specificity': [2, 1, 3, 0, 2, 1],
                'blob_polarity': [0.5, -0.3, 0.8, -0.9, 0.2, -0.6],
                'blob_subjectivity': [0.6, 0.4, 0.7, 0.3, 0.5, 0.2],
                'vader_neg': [0.1, 0.8, 0.05, 0.9, 0.2, 0.7],
                'vader_neu': [0.3, 0.1, 0.2, 0.05, 0.4, 0.2],
                'vader_pos': [0.6, 0.1, 0.75, 0.05, 0.4, 0.1],
                'vader_compound': [0.7, -0.8, 0.9, -0.95, 0.3, -0.7],
                'stars': [5, 1, 5, 1, 3, 2],
                'num_reviews': [5, 2, 10, 1, 3, 4],
                'review_frequency': [1, 0, 2, 0, 1, 1],
                'rating_variance': [1.0, 0.5, 1.2, 0.0, 0.8, 0.9],
                'is_burst': [0, 0, 1, 0, 0, 0],
                'sentiment_star_diff': [0.1, 0.2, 0.05, 0.3, 0.4, 0.5],
                'is_extreme': [1, 1, 1, 1, 0, 0],
                'emotional_intensity': [3, 4, 4, 5, 2, 3]
            }
            features_list = [dummy_data]
            labels = [1, 0, 1, 0, 1, 0]

        feature_df = pd.DataFrame(features_list)
        feature_df = feature_df[self.feature_cols]
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.fit(feature_df, labels)
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.xgb_model, 'models/xgb_model.pkl')
        st.info(f"Trained and saved new model with {len(self.feature_cols)} features")
        
        importance = self.xgb_model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': self.feature_cols, 'Importance': importance})
        st.write("Feature Importance:", feature_importance.sort_values(by='Importance', ascending=False))

    def predict(self, text, stars, user_id="default"):
        features, translated_text = self.extract_features(text, stars, user_id)
        feature_df = pd.DataFrame([features])
        X = feature_df[self.feature_cols].values
        pred_proba = self.xgb_model.predict_proba(X)[0]
        prediction = self.xgb_model.predict(X)[0]
        compound_score = features['vader_compound']
        sentiment = "Positive" if compound_score >= 0.05 else "Negative" if compound_score <= -0.05 else "Neutral"

        return {
            'prediction': 'Genuine' if prediction == 0 else 'Fake',
            'confidence': max(pred_proba) * 100,
            'sentiment': sentiment,
            'sentiment_score': compound_score,
            'features': features,
            'translated_text': translated_text
        }

# Initialize detector
def load_detector():
    return FakeReviewDetector()

detector = load_detector()

# Function to generate voice output
def generate_voice_output(result, lang='en'):
    text = f"The review is {result['prediction']} with a confidence of {result['confidence']:.1f} percent. " \
           f"The sentiment is {result['sentiment']} with a score of {result['sentiment_score']:.2f}."
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_file = "output.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Voice output generation failed: {e}")
        return None

# Streamlit UI
def main():
    st.title(f"🔍 {translated_ui['title']}")
    st.markdown(translated_ui['description'])

    with st.spinner(translate_text("Checking datasets...", target_lang_code)):
        download_datasets()

    with st.expander(f"📝 {translated_ui['enter_review']}", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            default_review = "Absolutely loved the food here! The pasta was perfectly cooked, and the flavors were amazing. The staff was friendly and attentive, and the ambiance was cozy. A bit pricey, but worth every penny for the quality. Will definitely come back!"
            review_text = st.text_area(
                translated_ui['product_review'], 
                height=150, 
                placeholder=translate_text("Enter your review here...", target_lang_code),
                value=translate_text(default_review, target_lang_code)
            )
        with col2:
            star_rating = st.slider(translated_ui['star_rating'], 1, 5, 5)
            submit_btn = st.button(translated_ui['analyze_btn'])

    if submit_btn and review_text.strip():
        with st.spinner(translate_text("Analyzing review...", target_lang_code)):
            result = detector.predict(review_text, star_rating, user_id="default")
            user_history = detector.get_user_history(user_id="default")
            audio_file = generate_voice_output(
                result, 
                lang=target_lang_code if target_lang_code in ['en', 'es', 'fr', 'hi', 'zh-cn'] else 'en'
            )

        st.success(translated_ui['analysis_complete'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            badge_color = "green" if result['prediction'] == 'Genuine' else "red"
            badge_text = translated_ui['prediction_genuine'] if result['prediction'] == 'Genuine' else translated_ui['prediction_fake']
            st.markdown(f"""
                <div style="border: 2px solid {badge_color}; border-radius: 10px; padding: 20px; text-align: center;">
                    <h3 style="color: {badge_color}; margin: 0;">{badge_text}</h3>
                    <p style="font-size: 14px; margin: 5px 0 0;">{translated_ui['confidence']}: {result['confidence']:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            sentiment_color = "green" if result['sentiment'] == 'Positive' else "red" if result['sentiment'] == 'Negative' else "gray"
            st.markdown(f"""
                <div style="border: 2px solid {sentiment_color}; border-radius: 10px; padding: 20px; text-align: center;">
                    <h3 style="color: {sentiment_color}; margin: 0;">{translate_text(result['sentiment'], target_lang_code)} {translated_ui['sentiment']}</h3>
                    <p style="font-size: 14px; margin: 5px 0 0;">{translated_ui['sentiment_score']}: {result['sentiment_score']:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            rating_color = "green" if star_rating >= 4 else "red" if star_rating <= 2 else "orange"
            sentiment_match = translated_ui['matches_sentiment'] if (star_rating >= 4 and result['sentiment'] == 'Positive') else translated_ui['mismatch']
            st.markdown(f"""
                <div style="border: 2px solid {rating_color}; border-radius: 10px; padding: 20px; text-align: center;">
                    <h3 style="color: {rating_color}; margin: 0;">{translated_ui['user_rating']}: {star_rating} ★</h3>
                    <p style="font-size: 14px; margin: 5px 0 0;">{sentiment_match}</p>
                </div>
            """, unsafe_allow_html=True)

        # Voice output
        if audio_file:
            st.audio(audio_file, format="audio/mp3")
            if st.button(translated_ui['voice_output']):
                st.audio(audio_file, format="audio/mp3")

        tab1, tab2, tab3, tab4 = st.tabs([
            f"🌐 {translated_ui['language_tab']}",
            f"📈 {translated_ui['sentiment_tab']}",
            f"🧑‍💼 {translated_ui['history_tab']}",
            f"🔍 {translated_ui['features_tab']}"
        ])
        with tab1:
            lang = detector.detect_language(review_text)
            st.subheader(translated_ui['language_analysis'])
            col1, col2 = st.columns(2)
            with col1:
                st.metric(translated_ui['detected_language'], lang.upper())
                st.metric(translated_ui['translated'], "✅" if result['translated_text'] != review_text else translated_ui['not_needed'])
            with col2:
                wordcloud = WordCloud(width=400, height=200, background_color='white').generate(result['translated_text'])
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

        with tab2:
            st.subheader(translated_ui['sentiment_vs_rating'])
            fig = go.Figure()
            fig.add_trace(go.Indicator(mode="gauge+number", value=result['sentiment_score'], domain={'x': [0, 0.5], 'y': [0.5, 1]}, title={'text': translated_ui['sentiment_score']}))
            fig.add_trace(go.Indicator(mode="number", value=star_rating, number={'suffix': " ★"}, domain={'x': [0.5, 1], 'y': [0.5, 1]}, title={'text': translated_ui['user_rating']}))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader(translated_ui['reviewer_history'])
            if not user_history:
                st.info(translated_ui['no_history'])
            else:
                df_history = pd.DataFrame(user_history)
                df_history['date'] = pd.to_datetime(df_history['date'])
                fig = px.scatter(df_history, x='date', y='stars', color='is_fake', size=np.abs(df_history['sentiment']) * 10 + 5, hover_data=['text'])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader(translated_ui['feature_breakdown'])
            feature_df = pd.DataFrame({'Feature': list(result['features'].keys()), 'Value': [round(v, 4) if isinstance(v, float) else v for v in result['features'].values()]})
            st.dataframe(feature_df.style.background_gradient(cmap='Blues'), hide_index=True)

    elif submit_btn:
        st.error(translated_ui['please_enter_review'])

if __name__ == "__main__":
    main()