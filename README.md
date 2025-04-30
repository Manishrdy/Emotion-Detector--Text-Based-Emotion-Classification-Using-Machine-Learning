# 🧠 Emotion Classification from Text using Machine Learning

This project focuses on classifying emotions from short text messages using various machine learning models. We used the GoEmotions dataset (subset of 6 emotion classes), trained and evaluated multiple models, and deployed the best-performing one — **SVM (Support Vector Machine)** — to a web app using Flask.

🔗 **Live Demo**: [Hugging Face Spaces Deployment](https://huggingface.co/spaces/martianthe/emotion-detector)

---

## 📌 Features

- Multi-class emotion detection (`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`)
- Model comparison: Logistic Regression, SVM, Random Forest, MLP
- Evaluation using 5-fold CV with weighted F1-score
- Confusion matrix and classification reports for model insights
- Web interface and REST API using Flask
- Deployed to Hugging Face Spaces

---

## 🚀 Best Model: Linear SVM

- **Accuracy**: 92%
- **Weighted F1-score**: 0.92
- **Macro F1-score**: 0.89
- **Well-classified**: sadness, joy, anger
- **Most confused**: surprise ↔ fear

---

## 🏗️ MLP Architecture (Alternative Model)

- Input Layer: TF-IDF dense features
- Hidden Layer 1: 512 neurons, ReLU + Dropout(0.5)
- Hidden Layer 2: 256 neurons, ReLU + Dropout(0.5)
- Output Layer: 6 neurons (Softmax activation)

---

## 📁 Project Structure

    emotion-detector/
    ├── templates/
    │   └── index.html                 # 🖥️ Frontend HTML for user input and results or intermediate files
    ├── tfidf_vectorizer.pkl           # 🔤 Saved TF-IDF vectorizer
    ├── logistic_regression_model.pkl  # 🤖 Trained Logistic Regression model
    ├── svm_model.pkl                  # 🧠 Best-performing SVM model
    ├── mlp_model.h5                   # 🔗 Trained MLP model (Keras)
    ├── app.py                         # 🚀 Main Flask app for UI and API
    ├── Dockerfile                     # 🐳 Docker container configuration
    ├── requirements.txt               # 📦 Python dependencies
    └── README.md                      # 📘 Project documentation (you’re here!)


---

## 📄 Dockerfile Reference

To build and run the project in a container:

```dockerfile
# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Set environment variables for NLTK
ENV NLTK_DATA=/tmp/nltk_data

# Run the app
CMD ["python", "app.py"]

#Build and run
docker build -t emotion-detector .
docker run -p 7860:7860 emotion-detector
```
---

## 📦 Requirements.txt Reference
```
flask
joblib
nltk
numpy
tensorflow
scikit-learn
```

---

## 🧪 How to Use
🔬 Local Setup
```
git clone https://github.com/your-username/emotion-classification.git
cd emotion-classification

# Create environment & install dependencies
pip install -r requirements.txt

# Run the app locally
python app.py

```
Open your browser at http://localhost:7860

🌐 Online Deployment
```
Try the live demo on Hugging Face Spaces:
https://huggingface.co/spaces/martianthe/emotion-detector
```
---

## 📚 Dataset
A simplified 6-class subset of the original GoEmotions dataset, developed by Google Research.

---

## 📈 Evaluation Tools
5-fold Cross Validation (weighted F1)

Confusion Matrices (normalized + raw)

Classification Reports per model

Training/validation loss graphs for MLP

---

## 📌 Future Improvements
Add BERT or transformer-based models

Train using full GoEmotions label set (27+ classes)

Improve UI with real-time prediction updates

Deploy with GPU acceleration for faster inference

---