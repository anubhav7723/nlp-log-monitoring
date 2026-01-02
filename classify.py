import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Models (ONCE)
# ----------------------------
bert_model = SentenceTransformer("models/bert_model")

classifier = joblib.load("models/log_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

CONFIDENCE_THRESHOLD = 0.70

# ----------------------------
# Classification Function
# ----------------------------
def classify_log(log_message: str):
    """
    Returns:
        label (str)
        confidence (float)
        severity (str)
    """

    # 1️⃣ Generate BERT embedding
    embedding = bert_model.encode([log_message])

    # 2️⃣ Predict probabilities
    probs = classifier.predict_proba(embedding)[0]
    max_prob = np.max(probs)
    pred_class = np.argmax(probs)

    # 3️⃣ Decode label
    label = label_encoder.inverse_transform([pred_class])[0]

    # 4️⃣ Confidence check
    if max_prob < CONFIDENCE_THRESHOLD:
        label = "Unknown"

    # 5️⃣ Simple severity mapping
    severity = map_severity(label)

    return {
        "label": label,
        "confidence": float(max_prob),
        "severity": severity
    }


def map_severity(label: str):
    if label in ["Database Error", "System Crash", "OutOfMemory"]:
        return "Critical"
    elif label in ["Timeout", "Connection Error"]:
        return "Error"
    elif label in ["Warning"]:
        return "Warning"
    else:
        return "Info"
