from flask import Flask, request, jsonify, Response
from bs4 import BeautifulSoup

import joblib
import requests
import spacy
import pandas as pd

import re
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

nlp = spacy.load("ru_core_news_sm")

vectorizer_tfidf = joblib.load("models/vectorizer_tfidf.joblib")
svc_model = joblib.load("models/SVC.joblib")
regression_model = joblib.load("models/Regression.joblib")

with open("names.json", 'r', encoding='utf-8') as f:
    names = json.load(f)


@app.route("/svc", methods=["POST", "GET"])
def multiclass_svc():
    url = request.args.get("url", default="_", type=str)

    if url == "_":
        return "url argument wasn't passed"
    else:
        data = website_text(url, svc_model)
        response = format_response(data)

        return response


@app.route("/regression", methods=["POST", "GET"])
def multiclass_regression():
    url = request.args.get("url", default="_", type=str)

    if url == "_":
        return "url argument wasn't passed"
    else:
        data = website_text(url, regression_model)
        response = format_response(data)

        return response


def format_response(data):
    json_data = json.dumps(data, ensure_ascii=False)
    json_bytes = json_data.encode('utf-8')

    response = Response(response=json_bytes, status=200, mimetype='application/json')
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


def website_text(url: str, model):
    try:
        response = requests.get(url)
        html_text = response.text
        soup = BeautifulSoup(html_text, "html.parser")

        text: str = soup.getText()
        text: str = re.sub(r'\s+', ' ', text)

        sentences = get_sentences(text)
        info = []

        for i in sentences:
            phrase_info, prob = classify_phrases(i, model)
            if phrase_info is not None:
                category = f'{names[str(phrase_info.tolist()[0])]} probability: {prob}'
                info.append([category, i])
        return info
    except requests.exceptions.ConnectionError as e:
        return


def get_sentences(text: str):
    doc = nlp(text)
    sentences = []

    for i in doc.sents:
        sentences += i.text.split(',')

    return sentences


def classify_phrases(phrase: str, model):
    vectorized = vectorizer_tfidf.transform([phrase])

    phrase_info = model.predict(vectorized)
    prob = model.predict_proba(vectorized)[0]
    if prob[phrase_info[0]] < 0.5:
        return None, None
    else:
        return phrase_info, prob[phrase_info[0]]


if __name__ == '__main__':
    app.run()
