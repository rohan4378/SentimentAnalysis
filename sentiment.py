import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Create 'model' directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Ensure NLTK data is downloaded
try:
    # Check for stopwords
    nltk.data.find('corpora/stopwords')
except LookupError: # Catch LookupError for resource not found
    print("Downloading stopwords...")
    nltk.download('stopwords')

try:
    # Check for wordnet
    nltk.data.find('corpora/wordnet')
except LookupError: # Catch LookupError for resource not found
    print("Downloading wordnet...")
    nltk.download('wordnet')


# Load the dataset
# Setting header=None, providing column names, and using usecols=[0, 1]
# Keeping default comma separator, engine='python', and on_bad_lines='skip'
df = pd.read_csv(
    r"c:\Users\IMDB Dataset.csv\IMDB Dataset.csv",
    header=None,          # Indicate no header row
    names=['review', 'sentiment'], # Provide column names manually
    usecols=[0, 1],       # Explicitly use only the first two columns (0-indexed)
    on_bad_lines='skip', # Attempt to skip malformed lines
    engine='python'    # Explicitly use the Python engine
)

# --- Diagnostic step: Print the actual column names ---
print("Columns after reading CSV:", df.columns)
# --- End of diagnostic step ---

# The columns should now be correctly named 'review' and 'sentiment'
# We can directly select and proceed if they exist

if 'review' in df.columns and 'sentiment' in df.columns:
    # Select and make a copy to avoid SettingWithCopyWarning later
    # This step is redundant now with usecols and names, but kept for clarity.
    df = df[['review', 'sentiment']].copy()
    print("\nDataFrame head after successful reading:")
    print(df.head())

    # Define clean_text function
    def clean_text(text):
        # Ensure the input is a string
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)  # Remove HTML
        text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
        return text

    # Clean text
    df['clean_review'] = df['review'].apply(clean_text)

    # Convert sentiment labels to numerical format using LabelEncoder
    le = LabelEncoder()
    # Handle potential NaN values in the sentiment column before encoding
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'].astype(str).str.strip().str.lower().fillna('')) # Ensure string, strip, lower, handle NaN

    # Split data - now using the encoded sentiment column for the target
    X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['sentiment_encoded'], test_size=0.2, random_state=42)

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train logistic regression
    model = LogisticRegression()
    # Fit the model using the numerical y_train
    model.fit(X_train_tfidf, y_train)
    # Predict and evaluate
    y_pred = model.predict(X_test_tfidf)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # If you want the report with 'positive'/'negative' labels, you'd need to
    # inverse_transform the predictions or the true labels back.

else:
    print("\nError: Required columns 'review' and 'sentiment' were not created after reading the CSV.")
    print("Please inspect the CSV file manually to understand its structure.")
    # Raise a final error if columns are still missing
    raise KeyError("Failed to read CSV with expected 'review' and 'sentiment' columns after trying various parameters.")
import pickle

# Save the vectorizer and model
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
      
