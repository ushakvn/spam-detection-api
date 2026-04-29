from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI()

# ------------------ TRAINING DATA ------------------

texts = [
    "Win money now",
    "Claim your prize",
    "Hello friend how are you",
    "Let's meet tomorrow",
    "Free lottery ticket",
    "Important meeting today"
]

labels = [1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# ------------------ MODEL ------------------

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# ------------------ INPUT MODEL ------------------

class TextInput(BaseModel):
    text: str

# ------------------ API ------------------

@app.get("/")
def home():
    return {"message": "Spam Detection API Running"}

@app.post("/predict")
def predict(data: TextInput):
    transformed_text = vectorizer.transform([data.text])
    prediction = model.predict(transformed_text)

    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return {"prediction": result}