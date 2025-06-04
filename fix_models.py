import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

print('Creating compatible models...')

# Create model directory
os.makedirs('model', exist_ok=True)

# Create sample training data
sample_reviews = [
    'This movie is fantastic and amazing',
    'Terrible film, waste of time',
    'Great acting and storyline',
    'Boring and poorly made',
    'Excellent cinematography',
    'Very disappointing movie'
]
sample_sentiments = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Create and train vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vectorized = vectorizer.fit_transform(sample_reviews)

# Create and train model
model = LogisticRegression(random_state=42)
model.fit(X_vectorized, sample_sentiments)

# Save models with current numpy version
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Models created and saved successfully!')

# Test loading
try:
    with open('model/vectorizer.pkl', 'rb') as f:
        test_vectorizer = pickle.load(f)
    with open('model/sentiment_model.pkl', 'rb') as f:
        test_model = pickle.load(f)
    print('Models load successfully - compatibility confirmed!')
except Exception as e:
    print(f'Error loading models: {e}')
