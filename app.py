from flask import Flask, request, jsonify, render_template
import joblib
import string
import nltk
from nltk.corpus import stopwords
import pickle
import os
nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model/emotion_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

emotion_emoji_map = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/home')
def home():
    return render_template('index.html')

# Route to test API with raw text
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"emotion": "Please enter text!"})

    preprocessed_text = preprocess(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]
    proba = model.predict_proba(vectorized_text)[0]

    # Get labels and convert probabilities to dictionary
    labels = model.classes_
    probas_dict = {label: float(prob) for label, prob in zip(labels, proba)}

    emoji = emotion_emoji_map.get(prediction, "")
    return jsonify({
        "emotion": prediction,
        "emoji": emoji,
        "probabilities": probas_dict
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)




