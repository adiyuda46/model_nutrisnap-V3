# app.py
import os
import io
import base64
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

#init Flask
app = Flask(__name__)


#n get file model & lebel class
MODEL_PATH  = "my_model_2.keras"
LABELS_PATH = "class_names.txt"

# Load model
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Load clas per row  1 1
if not os.path.isfile(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

# set ukuran model (gambar input)
try:
    _, height, width, _ = model.input_shape   
except Exception:
    height, width = 224, 224                
INPUT_SHAPE = (height, width)

# ubah Base64 → bytes → Tensor
def preprocess_image_from_bytes(image_bytes: bytes) -> tf.Tensor:
    """
    - Buka gambar dari bytes, konversi ke RGB
    - Resize ke INPUT_SHAPE
    - Normalisasi ke [0, 1]
    - Tambah dimensi batch
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SHAPE)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, 0)          # (1, H, W, 3)
    return tf.convert_to_tensor(img_arr, dtype=tf.float32)

def decode_base64_image(b64_string: str) -> bytes:
    """
    Menerima string Base64 yang mungkin berada dalam format:
        "data:image/jpeg;base64,/9j/4AAQ..."
    atau hanya bagian setelah koma.
    """
 # Jika mengandung header "data:*;base64," kita buang bagian depan
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,")[1]
    try:
        return base64.b64decode(b64_string)
    except Exception as exc:
        raise ValueError("Base64 decoding error: " + str(exc))

# Endpoint /predict  (POST)
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body:
    {
        "file": "data:image/jpeg;base64,/9j/4AAQ..."
    }

    Response:
    {
        "label": "nama_kelas",
        "confidence": 0.9876
    }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if "file" not in data:
        return jsonify({"error": "Field 'file' tidak ditemukan di JSON"}), 400

    try:
        # 1️⃣ Decode base64 → raw bytes
        image_bytes = decode_base64_image(data["file"])

        # 2️⃣ Pre‑process menjadi Tensor
        img_tensor = preprocess_image_from_bytes(image_bytes)

        # 3️⃣ Prediksi
        preds = model.predict(img_tensor)

        # 4️⃣ Softmax bila perlu
        if preds.ndim == 2:
            probs = tf.nn.softmax(preds, axis=1).numpy()[0]
        else:
            probs = preds[0]

        top_idx = int(np.argmax(probs))
        label = class_names[top_idx]
        confidence = float(probs[top_idx])

        return jsonify({
            "label": label,
            "confidence": round(confidence, 4)
        })
    except ValueError as ve:
        # Kesalahan decoding Base64
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Untuk debugging produksi bisa log ke file, di sini cukup kirim ke client
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Jalankan server
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # debug=False untuk produksi; aktifkan debug=True saat development
    app.run(host="0.0.0.0", port=port, debug=False)



