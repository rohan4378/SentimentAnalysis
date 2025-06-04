from flask import Flask, request, render_template
import pickle
import re
import string
print("Starting sentiment app...")

app = Flask(__name__)

# Load model and vectorizer
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    return render_template('index.html', review=review, prediction=sentiment)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8081)
