# 🚀 Spam Detection API

## 📌 Project Overview
This project is a Machine Learning API built using FastAPI that classifies text messages as Spam or Not Spam.

## 🧠 Technologies Used
- FastAPI
- Scikit-learn
- NLP (CountVectorizer)
- Naive Bayes

## ⚙️ How It Works
User → API → ML Model → Prediction

## ▶️ Run Locally
uvicorn main:app --reload

## 📥 Example Input
{
  "text": "Win a free lottery now"
}

## 📤 Output
{
  "prediction": "Spam"
}