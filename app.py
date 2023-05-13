from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

clf = joblib.load("naive_bayes_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    review_description = request.json.get("review_description", "")

    if not review_description:
        return jsonify({"error": "Review description is required"}), 400

    X = vectorizer.transform([review_description])

    prediction = clf.predict(X)

    return jsonify({"variety": prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
