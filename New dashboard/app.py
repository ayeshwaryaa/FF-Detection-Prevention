import os
import io
import base64
import numpy as np
import joblib
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Import your logic
from fire_detection_logic import FeatureExtractor, get_transform, sliding_window, non_max_suppression
from fuzzy_logic_module import FuzzyFireSystem

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# 1. LOAD MODELS SAFELY
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Initialize all as None first
svm_model = None
svm_scaler = None
catboost_model = None
baselines = None

try:
    svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm_model.joblib'))
    print("✅ SVM model loaded")
except Exception as e:
    print(f"❌ Failed to load SVM model: {e}")

try:
    svm_scaler = joblib.load(os.path.join(MODEL_DIR, 'svm_scaler.joblib'))
    print("✅ Scaler loaded")
except Exception as e:
    print(f"❌ Failed to load scaler: {e}")

# Feature extractor
try:
    extractor = FeatureExtractor()
    print("✅ Feature extractor initialized")
except Exception as e:
    print(f"❌ Feature extractor failed: {e}")
    extractor = None

# CatBoost + baselines (FIXED PATH ISSUE)


try:
    catboost_model = joblib.load('F:/MAJOR/fire_model_catboost.pkl')
    print("✅ CatBoost model loaded")
except Exception as e:
    print(f"❌ Failed to load CatBoost model: {e}")

try:
    baselines = joblib.load('F:/MAJOR/monthly_baselines.pkl')
    print("✅ Baselines loaded")
except Exception as e:
    print(f"❌ Failed to load baselines: {e}")

# Fuzzy system
try:
    fuzzy_system = FuzzyFireSystem()
    print("✅ Fuzzy system ready")
except Exception as e:
    print(f"❌ Fuzzy system failed: {e}")
    fuzzy_system = None


# ==========================================
# 2. MODEL VALIDATION
# ==========================================
def check_models():
    try:
        assert svm_model is not None, "SVM model not loaded"
        assert svm_scaler is not None, "Scaler not loaded"
        assert extractor is not None, "Feature extractor missing"
        assert catboost_model is not None, "CatBoost model not loaded"
        assert baselines is not None, "Baselines not loaded"
        assert fuzzy_system is not None, "Fuzzy system missing"

        print("🎉 ALL MODELS VERIFIED SUCCESSFULLY!")

    except Exception as e:
        print(f"❌ MODEL CHECK FAILED: {e}")
        exit(1)


def test_models():
    try:
        # Dummy test for CatBoost
        dummy = np.zeros(11)
        catboost_model.predict_proba([dummy])

        # Dummy fuzzy test
        fuzzy_system.calculate_risk(80, 10)

        print("🚀 Models are WORKING correctly!")

    except Exception as e:
        print(f"❌ MODEL EXECUTION FAILED: {e}")


# Run checks at startup
check_models()
test_models()


# ==========================================
# 3. HEALTH CHECK API
# ==========================================
@app.route('/health', methods=['GET'])
def health_check():
    try:
        status = {
            "svm_model": svm_model is not None,
            "svm_scaler": svm_scaler is not None,
            "feature_extractor": extractor is not None,
            "catboost_model": catboost_model is not None,
            "baselines": baselines is not None,
            "fuzzy_system": fuzzy_system is not None
        }

        return jsonify({
            "status": "success",
            "models": status
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ==========================================
# 4. DETECTION ENDPOINT
# ==========================================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

        file = request.files['file']
        filename = file.filename.lower()
        if filename.endswith('.png'):
            ext = '.png'
            mime = 'image/png'
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            ext = '.jpg'
            mime = 'image/jpeg'
        elif filename.endswith('.webp'):
            ext = '.webp'
            mime = 'image/webp'
        else:
            ext = '.jpg'  # fallback
            mime = 'image/jpeg'
        img_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_np = np.array(img_pil)

        WINDOW_SIZE = (128, 128)
        STEP_SIZE = 32
        NMS_THRESHOLD = 0.3

        raw_boxes = []
        confidences = []
        transform = get_transform()

        for (x, y, patch) in sliding_window(img_np, WINDOW_SIZE, STEP_SIZE):
            if patch.shape[0] != WINDOW_SIZE[1] or patch.shape[1] != WINDOW_SIZE[0]:
                continue

            patch_pil = Image.fromarray(patch)
            patch_tensor = transform(patch_pil).unsqueeze(0)

            features = extractor.extract(patch_tensor)
            features_scaled = svm_scaler.transform(features)

            probs = svm_model.predict_proba(features_scaled)[0]
            fire_prob = float(probs[1])

            if fire_prob > 0.7:
                raw_boxes.append([x, y, WINDOW_SIZE[0], WINDOW_SIZE[1]])
                confidences.append(fire_prob)

        final_boxes = non_max_suppression(raw_boxes, confidences, NMS_THRESHOLD)

        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for (x, y, w, h) in final_boxes:
            # default color (red)
            color = (0, 0, 255)

            # OPTIONAL: try matching confidence (best effort)
            for i, box in enumerate(raw_boxes):
                if box[0] == x and box[1] == y:
                    conf = confidences[i]

                    if conf > 0.85:
                        color = (0, 0, 255)      # 🔴 High
                    elif conf > 0.75:
                        color = (0, 165, 255)    # 🟠 Medium
                    else:
                        color = (0, 255, 255)    # 🟡 Low
                    break

            cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)

        _, buffer = cv2.imencode(ext, img_cv)
        img_str = base64.b64encode(buffer).decode('utf-8')

        is_fire = len(final_boxes) > 0
        max_conf = max(confidences) if confidences else 0.0

        return jsonify({
    'status': 'success',
    'fire_detected': is_fire,
    'confidence': f"{max_conf*100:.1f}%",
    'image_data': img_str,
    'mime_type': mime,   # ✅ ADD THIS
    'box_count': len(final_boxes)
})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==========================================
# 5. PREVENTION ENDPOINT
# ==========================================
@app.route('/predict_fire', methods=['POST'])
def predict_fire():
    try:
        data = request.get_json()

        max_t = float(data.get('max_temp', 0))
        min_t = float(data.get('min_temp', 0))
        wind = float(data.get('wind_speed', 0))
        rain = float(data.get('rain', 0))
        lag_rain = float(data.get('lagged_rain', 0))
        month = int(data.get('month', 1))

        temp_range = max_t - min_t
        ratio = wind / (max_t if max_t != 0 else 1)

        avg_t = baselines['MAX_TEMP'].get(month, 70)
        avg_w = baselines['AVG_WIND_SPEED'].get(month, 10)

        temp_anomaly = max_t - avg_t
        wind_anomaly = wind - avg_w

        explanations = []

        if temp_anomaly > 5:
            explanations.append("Temperature is significantly above normal")

        if wind_anomaly > 5:
            explanations.append("High wind speed may spread fire quickly")

        if rain < 1:
            explanations.append("Low rainfall indicates dry conditions")

        if lag_rain < 1:
            explanations.append("Recent days were dry")

        if not explanations:
            explanations.append("Conditions are within safe limits")
        fire_power = max_t * wind

        features = [
            max_t, min_t, wind, rain, lag_rain,
            temp_range, ratio, month,
            temp_anomaly, wind_anomaly, fire_power
        ]

        ml_prob = catboost_model.predict_proba([np.array(features)])[0][1] * 100
        fuzzy_score = fuzzy_system.calculate_risk(max_t, wind)

        final_score = (0.7 * ml_prob) + (0.3 * fuzzy_score)
        status = "HIGH RISK" if final_score > 50 else "SAFE"

        return jsonify({
        'status': 'success',
        'prediction': status,
        'probability': round(final_score, 2),
        'details': {
            'ml_confidence': round(ml_prob, 1),
            'fuzzy_verification': round(fuzzy_score, 1)
        },
        'explanations': explanations   # 🔥 NEW
    })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/generate_graphs', methods=['GET'])
def generate_graphs():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        # ---- Dummy test data (replace if you want real dataset) ----
        X = np.random.rand(100, 11)
        y_true = np.random.randint(0, 2, 100)

        ml_scores = []
        fuzzy_scores = []
        final_scores = []

        for i in range(len(X)):
            features = X[i]

            ml = catboost_model.predict_proba([features])[0][1] * 100
            fuzzy = fuzzy_system.calculate_risk(features[0], features[2])
            final = (0.7 * ml) + (0.3 * fuzzy)

            ml_scores.append(ml)
            fuzzy_scores.append(fuzzy)
            final_scores.append(final)

        # ---- ROC Curve ----
        fpr, tpr, _ = roc_curve(y_true, np.array(final_scores)/100)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        plt.plot([0,1],[0,1],'--')
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("roc_curve.png")
        plt.close()

        # ---- PR Curve ----
        precision, recall, _ = precision_recall_curve(y_true, np.array(final_scores)/100)

        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")
        plt.savefig("pr_curve.png")
        plt.close()

        # ---- ML vs Fuzzy vs Final ----
        plt.figure()
        plt.plot(ml_scores, label="ML")
        plt.plot(fuzzy_scores, label="Fuzzy")
        plt.plot(final_scores, label="Final")
        plt.legend()
        plt.title("ML vs Fuzzy vs Final")
        plt.savefig("comparison.png")
        plt.close()

        return jsonify({
            "status": "success",
            "message": "Graphs generated successfully!"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
# ==========================================
# 6. RUN SERVER
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')