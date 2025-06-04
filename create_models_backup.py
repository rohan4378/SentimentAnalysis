import pandas as pd
import numpy as np
import re
import string
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

print("Starting model training...")

# Create 'model' directory if it doesn't exist
os.makedirs('model', exist_ok=True)

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    return text

# Try to load IMDB dataset, fallback to sample data if not available
try:
    print("Attempting to load IMDB dataset...")
    # Try multiple possible paths for the dataset
    possible_paths = [
        "/app/IMDB Dataset.csv",
        "IMDB Dataset.csv",
        r"c:\Users\IMDB Dataset.csv\IMDB Dataset.csv"
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded dataset from: {path}")
            break
        except:
            continue
    
    if df is None:
        raise FileNotFoundError("IMDB dataset not found")
        
    # Use the actual dataset
    print(f"Dataset shape: {df.shape}")
    print("Dataset columns:", df.columns.tolist())
    
    # Handle different possible column names
    if 'review' in df.columns and 'sentiment' in df.columns:
        reviews = df['review']
        sentiments = df['sentiment']
    elif len(df.columns) >= 2:
        reviews = df.iloc[:, 0]  # First column
        sentiments = df.iloc[:, 1]  # Second column
    else:
        raise ValueError("Cannot identify review and sentiment columns")
    
    # Clean the data
    print("Cleaning text data...")
    clean_reviews = reviews.apply(clean_text)
    
    # Encode sentiments
    le = LabelEncoder()
    sentiment_encoded = le.fit_transform(sentiments.astype(str).str.strip().str.lower())
    
    print(f"Using {len(clean_reviews)} real reviews for training")

except Exception as e:
    print(f"Could not load IMDB dataset: {e}")
    print("Using expanded sample data instead...")
    
    # Expanded sample data with more variety
    sample_reviews = [
        "This movie was absolutely fantastic! Great acting and plot.",
        "Terrible film, waste of time. Poor acting and boring story.",
        "Amazing cinematography and wonderful performances by all actors.",
        "One of the worst movies I have ever seen. Completely disappointing.",
        "Brilliant storytelling and excellent direction. Highly recommended!",
        "Awful script and terrible execution. Not worth watching.",
        "Outstanding film with great character development and plot twists.",
        "Boring and predictable. The worst movie of the year.",
        "Incredible acting and beautiful scenes. A masterpiece!",
        "Poorly made with bad acting. Complete disappointment.",
        "Excellent movie with great special effects and story.",
        "Terrible plot and poor character development. Very bad.",
        "Wonderful film with amazing performances. Love it!",
        "Worst movie ever made. Completely awful experience.",
        "Great entertainment value and excellent cinematography.",
        "Bad acting and boring storyline. Not recommended.",
        "Fantastic movie with incredible visual effects and story.",
        "Poor direction and terrible script. Very disappointing.",
        "Amazing film with great actors and beautiful soundtrack.",
        "Awful movie with bad plot and poor execution."
    ]
    
    sample_sentiments = [
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative"
    ]
    
    clean_reviews = pd.Series([clean_text(review) for review in sample_reviews])
    
    # Encode sentiments
    le = LabelEncoder()
    sentiment_encoded = le.fit_transform(sample_sentiments)
    
    print(f"Using {len(clean_reviews)} sample reviews for training")

# Split the data
print("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    clean_reviews, sentiment_encoded, test_size=0.2, random_state=42
)

# Vectorize with TF-IDF
print("Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression model
print("Training logistic regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
print("Evaluating model performance...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test with a few examples
print("\nTesting model with sample predictions:")
test_texts = [
    "This is a great movie!",
    "This movie is terrible and boring.",
    "Amazing film with excellent acting."
]

for text in test_texts:
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment_label = "Positive" if prediction == 1 else "Negative"
    print(f"Text: '{text}' -> Prediction: {sentiment_label}")

# Save the models
print("\nSaving models...")
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the label encoder for reference
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Models saved successfully!")
print("Vectorizer saved to: model/vectorizer.pkl")
print("Sentiment model saved to: model/sentiment_model.pkl")
print("Label encoder saved to: model/label_encoder.pkl")
