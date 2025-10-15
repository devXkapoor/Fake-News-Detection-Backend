# Flask backend for AI Fake News Detector (integrated with Streamlit frontend)
# Runs locally, supports fallback without torch/transformers, and uses simple /predict endpoint.

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from typing import Tuple

# --- Optional ML libraries ---
TORCH_AVAILABLE = True
TRANSFORMERS_AVAILABLE = True
try:
    import torch
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ModuleNotFoundError:
    TRANSFORMERS_AVAILABLE = False

# --- Config ---
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-base-openai-detector")  # same as frontend
DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

# --- Load model (if available) ---
tokenizer = None
model = None
MODEL_LOADED = False

if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    try:
        print(f"Loading model: {MODEL_NAME} on {DEVICE}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        MODEL_LOADED = True
        print("Model loaded successfully.")
    except Exception as e:
        print("⚠️ Failed to load model:", e, file=sys.stderr)
else:
    print("⚠️ torch/transformers not available — using fallback keyword heuristic.")

# --- Fallback keyword heuristic ---
SUSPICIOUS_KEYWORDS = [
    "shocking", "you won't believe", "breaking", "exclusive", "shocker", "unbelievable",
    "click here", "must see", "viral", "claims", "alleged", "hoax", "fake", "conspiracy",
    "sources say", "reports say", "leaked", "sensational"
]

def heuristic_predict(text: str) -> Tuple[str, float]:
    lower = text.lower()
    hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lower)
    score = min(0.5 + 0.12 * hits, 0.99)
    label = "FAKE" if hits >= 1 else "REAL"
    return label, round(score, 4)

# --- Flask app setup ---
app = Flask(__name__)
CORS(app)  # Allow Streamlit to make API requests

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "device": DEVICE,
        "torch_available": TORCH_AVAILABLE,
        "transformers_available": TRANSFORMERS_AVAILABLE
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        if MODEL_LOADED:
            # Use transformer model
            print("model loaded!")
            enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                outputs = model(**enc)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                top_idx = torch.argmax(probs).item()
                score = float(probs[top_idx])
                label_map = getattr(model.config, "id2label", {0: "REAL", 1: "FAKE"})
                label = label_map.get(top_idx, "REAL")
        else:
            # Use fallback
            print("getting back to heuristic")
            label, score = heuristic_predict(text)

        return jsonify({
            "label": label,
            "score": score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Local test route (optional) ---
@app.route("/test", methods=["GET"])
def test():
    examples = [
        "Breaking: Scientists discover cure for aging — click here to read!",
        "Government launches new AI policy for schools.",
    ]
    results = []
    for ex in examples:
        label, score = heuristic_predict(ex)
        results.append({"text": ex, "label": label, "score": score})
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Flask Fake News API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# -------------------------
# ✅ This backend is ready for your Streamlit frontend.
# To connect:
# 1. Run this Flask backend:  python app.py
# 2. In Streamlit, replace model pipeline with an API request:
#    import requests
#    resp = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
#    result = resp.json()
#    label, score = result["label"], result["score"]
# 3. Display these results in your Streamlit UI.
# -------------------------
