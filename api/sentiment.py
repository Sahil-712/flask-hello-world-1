import re
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

app = Flask(__name__)

# Load pre-trained BERT model once at startup
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Emoji sentiment dictionary
emoji_sentiment = {
    "😊": "positive", "😃": "positive", "😁": "positive", "😆": "positive", "😂": "positive", "🤣": "positive",
    "😍": "positive", "🥰": "positive", "😘": "positive", "😗": "positive", "😚": "positive", "😙": "positive",
    "👍": "positive", "👏": "positive", "💖": "positive", "💞": "positive", "💕": "positive", "💓": "positive",
    "💗": "positive", "💘": "positive", "💝": "positive", "🎉": "positive", "🥳": "positive", "💯": "positive",
    "😢": "negative", "😭": "negative", "😡": "negative", "😠": "negative", "💔": "negative", "😞": "negative",
    "😖": "negative", "😣": "negative", "😩": "negative", "😫": "negative", "😤": "negative", "👎": "negative",
    "🙁": "negative", "☹️": "negative", "😕": "negative", "🤬": "negative", "😨": "negative", "😰": "negative",
    "😱": "negative", "😒": "negative", "😔": "negative", "🥺": "negative", "😑": "negative", "😶": "negative"
}

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    sentiment_score = torch.argmax(probs, dim=-1).item()

    if sentiment_score in [0, 1]:
        return "negative"
    elif sentiment_score == 2:
        return "neutral"
    else:
        return "positive"

def extract_emoji_sentiment(text):
    for char in text:
        if char in emoji_sentiment:
            return emoji_sentiment[char]
    return None

def classify_sentiment(text):
    text_sentiment = analyze_sentiment(text)
    emoji_sent = extract_emoji_sentiment(text)

    if emoji_sent:
        if text_sentiment == "positive" and emoji_sent == "positive":
            return "Positive"
        elif text_sentiment == "negative" and emoji_sent == "negative":
            return "Negative"
        elif (text_sentiment == "positive" and emoji_sent == "negative") or \
             (text_sentiment == "negative" and emoji_sent == "positive"):
            return "Contradictory"
    
    return text_sentiment.capitalize()

@app.route("/sentiment", methods=["POST"])
def sentiment_api():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    sentiment = classify_sentiment(text)
    return jsonify({"sentiment": sentiment})

# For Vercel compatibility
def handler(environ, start_response):
    return app(environ, start_response)
