from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

texts = [
    "Win money now",
    "Claim your prize",
    "Hello friend",
    "Let's meet tomorrow",
    "Free lottery ticket",
    "Important meeting today"
]

labels = [1, 1, 0, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")