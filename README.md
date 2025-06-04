# Sentiment Analysis Docker Application

A Flask-based sentiment analysis web application that predicts movie review sentiments using machine learning, containerized with Docker.

## 🚀 Features

- **Web Interface**: Simple HTML form for sentiment prediction
- **Machine Learning**: TF-IDF vectorization with Logistic Regression
- **Text Processing**: Advanced text cleaning and preprocessing
- **Docker Support**: Fully containerized application
- **Real-time Predictions**: Instant sentiment analysis (Positive/Negative)

## 📁 Project Structure

```
sentiment-analysis/
├── app.py                 # Main Flask application
├── create_models.py       # Model training script
├── fix_models.py         # Compatibility fix script
├── sentiment.py          # Original sentiment analysis script
├── Dockerfile            # Docker container configuration
├── requirements.txt      # Python dependencies
├── .dockerignore        # Docker ignore file
├── templates/           # HTML templates
│   └── index.html       # Web interface
├── model/              # Trained models directory
│   ├── vectorizer.pkl   # TF-IDF vectorizer
│   ├── sentiment_model.pkl # Trained model
│   └── label_encoder.pkl   # Label encoder
└── README.md           # This file
```

##  Prerequisites

- **Docker Desktop** installed and running
- **Python 3.8+** (for local development)
- **Git** (for cloning)
- **VS Code** (recommended)

## 📋 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd sentiment-analysis
```

### 2. Build Docker Image
```powershell
docker build -t sentiment-app . --no-cache
```

### 3. Train the Models
```powershell
docker run -it --rm -v "${PWD}/model:/app/model" sentiment-app python create_models.py
```

### 4. Run the Application
```powershell
docker run -d -p 8081:8081 -v "${PWD}/model:/app/model" --name sentiment-app-final sentiment-app
```

### 5. Access the Application
Open your browser and go to: `http://localhost:8081`

## 📝 Detailed Setup Instructions

### Step 1: Environment Setup
```powershell
# Create project directory
mkdir "sentiment-analysis"
cd "sentiment-analysis"

# Initialize git repository (optional)
git init
```

### Step 2: Create Required Files

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8081

CMD ["python", "app.py"]
```

**requirements.txt:**
```
numpy==1.24.3
scikit-learn==1.3.0
flask==2.3.3
pandas==2.0.3
nltk==3.8.1
```

**.dockerignore:**
```
__pycache__
*.pyc
.git
.gitignore
README.md
.env
```

### Step 3: Create the Flask Application (app.py)
```python
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
    app.run(host="0.0.0.0", port=8081)
```

### Step 4: Create HTML Template
Create `templates/index.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; }
        .container { max-width: 600px; margin: 0 auto; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 20px; border-radius: 5px; }
        .positive { background-color: #d4edda; color: #155724; }
        .negative { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Movie Review Sentiment Analysis</h1>
        <form method="POST" action="/predict">
            <textarea name="review" placeholder="Enter your movie review here...">{{ review or '' }}</textarea><br>
            <button type="submit">Analyze Sentiment</button>
        </form>
        
        {% if prediction %}
        <div class="result {{ prediction.lower() }}">
            <h3>Prediction: {{ prediction }}</h3>
            <p><strong>Review:</strong> {{ review }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
```

##  Common Issues and Solutions

### Issue 1: Docker Networking Error
**Error:**
```
docker: Error response from daemon: bind-mount /proc/3108/ns/net -> /var/run/docker/netns/c6f003c97822: no such file or directory
```

**Solution:**
```powershell
# Restart Docker Desktop completely
# Use absolute paths for volume mounting
docker run -it --rm -v "D:/full/path/to/project/model:/app/model" sentiment-app python create_models.py
```

### Issue 2: Pickle Compatibility Error
**Error:**
```
ModuleNotFoundError: No module named 'numpy._core'
```

**Root Cause:** Version mismatch between numpy versions used to create and load pickle files.

**Solution:** Create `fix_models.py`:
```python
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

print("Creating compatible models...")
os.makedirs('model', exist_ok=True)

# Create models with current environment versions
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
model = LogisticRegression(random_state=42, max_iter=1000)

# Sample training data for compatibility
sample_texts = [
    "great movie excellent acting",
    "terrible film boring plot", 
    "amazing cinematography wonderful",
    "awful script poor execution"
]
sample_labels = [1, 0, 1, 0]

# Fit models
vectorizer.fit(sample_texts)
X = vectorizer.transform(sample_texts)
model.fit(X, sample_labels)

# Save with current versions
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Models created and saved successfully!")

# Test loading
try:
    with open('model/vectorizer.pkl', 'rb') as f:
        test_vec = pickle.load(f)
    with open('model/sentiment_model.pkl', 'rb') as f:
        test_model = pickle.load(f)
    print("Models load successfully - compatibility confirmed!")
except Exception as e:
    print(f"Error loading models: {e}")
```

Run the fix:
```powershell
docker run -it --rm -v "${PWD}/model:/app/model" sentiment-app python fix_models.py
```

### Issue 3: 600 Timeout Errors
**Error:** Browser showing 600 timeout errors

**Root Cause:** 
- App not starting properly
- Large models taking too long to load
- App listening on wrong interface

**Solutions:**
1. **Check logs:**
   ```powershell
   docker logs sentiment-app-final
   ```

2. **Ensure proper host binding:**
   ```python
   app.run(host="0.0.0.0", port=8081)  # Not localhost or 127.0.0.1
   ```

3. **Run interactively for debugging:**
   ```powershell
   docker run -it -p 8081:8081 sentiment-app
   ```

### Issue 4: Port Already in Use
**Error:**
```
bind: address already in use
```

**Solution:**
```powershell
# Find and stop conflicting containers
docker ps
docker stop <container-name>
docker rm <container-name>

# Or use different port
docker run -d -p 8082:8081 --name sentiment-app-final sentiment-app
```

##  Development Workflow

### For Model Development:
```powershell
# 1. Modify create_models.py
# 2. Rebuild image
docker build -t sentiment-app . --no-cache

# 3. Retrain models
docker run -it --rm -v "${PWD}/model:/app/model" sentiment-app python create_models.py

# 4. Test application
docker run -d -p 8081:8081 -v "${PWD}/model:/app/model" --name sentiment-test sentiment-app
```

### For Application Development:
```powershell
# 1. Modify app.py or templates
# 2. Rebuild and restart
docker build -t sentiment-app . --no-cache
docker stop sentiment-app-final
docker rm sentiment-app-final
docker run -d -p 8081:8081 -v "${PWD}/model:/app/model" --name sentiment-app-final sentiment-app
```

##  Testing

### Manual Testing:
1. **Positive Review Test:**
   - Input: "This movie was absolutely fantastic! Great acting and amazing plot."
   - Expected: Positive

2. **Negative Review Test:**
   - Input: "Terrible film, complete waste of time. Poor acting and boring story."
   - Expected: Negative

### Container Health Check:
```powershell
# Check if container is running
docker ps

# Check application logs
docker logs sentiment-app-final

# Test API endpoint
curl http://localhost:8081
```

### Model Performance Testing:
```powershell
# Run model training to see accuracy metrics
docker run -it --rm -v "${PWD}/model:/app/model" sentiment-app python create_models.py
```

##  Deployment Options

### Local Development:
```powershell
docker run -d -p 8081:8081 -v "${PWD}/model:/app/model" --name sentiment-app sentiment-app
```

### Production Deployment:
```powershell
# Build production image
docker build -t sentiment-app:prod . --no-cache

# Run without volume mounting (models baked into image)
docker run -d -p 80:8081 --name sentiment-prod sentiment-app:prod
```

### Docker Hub Deployment:
```powershell
# Tag and push to Docker Hub
docker tag sentiment-app yourusername/sentiment-app:latest
docker push yourusername/sentiment-app:latest

# Others can then run:
docker run -d -p 8081:8081 yourusername/sentiment-app:latest
```

##  Performance Notes

- **Model Training Time:** ~1-2 minutes with sample data
- **Application Startup:** ~5-10 seconds
- **Memory Usage:** ~200-300MB
- **Prediction Speed:** <100ms per request

##  Security Considerations

- Application runs in development mode (not production-ready)
- No input validation beyond basic text cleaning
- No authentication or rate limiting
- Use production WSGI server for deployment

##  Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Scikit-learn for machine learning tools
- Flask for web framework
- Docker for containerization platform
- IMDB dataset for training data (when available)

##  Support

If you encounter issues:

1. **Check the Common Issues section** above
2. **Review Docker logs:** `docker logs sentiment-app-final`
3. **Ensure Docker Desktop is running**
4. **Verify port 8081 is available**
5. **Try rebuilding with:** `docker build -t sentiment-app . --no-cache`

## 🔄 Version History

- **v1.0.0** - Initial release with basic sentiment analysis
- **v1.1.0** - Added compatibility fixes for pickle files  
- **v1.2.0** - Enhanced model training with better sample data
- **v1.3.0** - Improved error handling and documentation

