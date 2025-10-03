from flask import Flask, request, jsonify
import joblib

import os

app=Flask(__name__)

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL_DIR=os.path.join(BASE_DIR, "../models")

nb_model=joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl"))
vectorizer=joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

@app.route("/")
def home():
    return "SMS Spam Classifier API is running!"

@app.route("/predict", methods = ["POST"])
def predict():
    #the input should be in JSON format
    #{ "message": "The text message in here"}
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "please provide a message in json"}), 400
    
    text = data["message"]
    text_tfidf = vectorizer.transform([text])
    prediction = nb_model.predict(text_tfidf)[0]

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)