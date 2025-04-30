from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import tensorflow as tf
import nltk
import os
import string

# ✅ Download and setup NLTK data (use /tmp for Hugging Face compatibility)
nltk_data_path = '/tmp/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)  # For lemmatizer
nltk.download('punkt_tab', download_dir=nltk_data_path)
from nltk.data import path
path.append(nltk_data_path)

# NLTK imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# ✅ Load your models
tfidf_vect = joblib.load('tfidf_vectorizer.pkl')
log_reg = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')
mlp_model = tf.keras.models.load_model('mlp_model.h5')
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([char for char in text if not char.isdigit()])
    text = text.strip()
    text = ' '.join(text.split())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def predict_emotion(text):
    processed = preprocess_text(text)
    tfidf = tfidf_vect.transform([processed])
    tfidf_dense = tfidf.toarray()
    pred_log = log_reg.predict(tfidf)[0]
    pred_svm = svm_model.predict(tfidf)[0]
    pred_mlp = np.argmax(mlp_model.predict(tfidf_dense), axis=1)[0]

    emotions = {
        0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'
    }

    return {
        'Logistic Regression': emotions[pred_log],
        'SVM': emotions[pred_svm],
        'MLP Neural Network': emotions[pred_mlp]
    }

# ✅ Web frontend route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        text = request.form.get('text')
        if text and text.strip():
            try:
                prediction = predict_emotion(text)
            except Exception as e:
                prediction = {'error': str(e)}
    return render_template('index.html', prediction=prediction)

# ✅ API route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        text = request.args.get('text', '')
    else:
        data = request.get_json()
        text = data.get('text', '') if data else ''

    if not text.strip():
        return jsonify({'error': 'No or empty text provided'}), 400

    try:
        predictions = predict_emotion(text)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
