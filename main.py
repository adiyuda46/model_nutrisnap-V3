# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         APP.PY — Flask Inference Server                    ║
# ║  Fix utama:                                                                 ║
# ║    1. Hapus double softmax (penyebab utama confidence ~0.11)                ║
# ║    2. Load MODEL_FT (hasil fine-tuning, bukan model_final)                  ║
# ║    3. Preprocessing eksplisit float32 + LANCZOS resize                      ║
# ║    4. Threshold confidence lebih rasional (0.5)                             ║
# ║    5. Debug log semua probabilitas per kelas                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import io
import base64
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

# ─── Init Flask ───────────────────────────────────────────────────────────────
app = Flask(__name__)

print("TensorFlow version:", tf.__version__)
print("Keras version     :", tf.keras.__version__)

# ─── Path Model & Label ───────────────────────────────────────────────────────
# ✅ FIX 1: Gunakan model hasil fine-tuning (MyModel_finetuned.keras),
#           bukan MyModel_final_epoch30.keras
#           MyModel_best   = hasil ModelCheckpoint fase 1 (frozen base)
#           MyModel_finetuned = hasil fine-tuning fase 2 → confidence lebih tinggi
MODEL_PATH  = "MyModel (3).keras"   # ← ganti dari MyModel_final_epoch30.keras
LABELS_PATH = "class_names.txt"

# ─── Load Model ───────────────────────────────────────────────────────────────
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model berhasil dimuat: {MODEL_PATH}")
print(f"   Input shape  : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")

# ─── Load Class Names ─────────────────────────────────────────────────────────
# ✅ FIX 2: class_names.txt harus disimpan dari train_generator.class_indices
#           (sudah dilakukan di Cell 7 training script)
#           Urutan baris di file ini = urutan index output softmax
if not os.path.isfile(LABELS_PATH):
    raise FileNotFoundError(f"[ERROR] Label tidak ditemukan: {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

print(f"✅ Kelas dimuat  : {class_names}")

# ─── Input Shape dari Model ───────────────────────────────────────────────────
try:
    _, height, width, _ = model.input_shape
except Exception:
    height, width = 224, 224

INPUT_SHAPE = (height, width)   # (224, 224)
print(f"   Input size   : {INPUT_SHAPE}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image_from_bytes(image_bytes: bytes) -> tf.Tensor:
    """
    Preprocessing gambar dari bytes agar konsisten dengan pipeline training.

    Pipeline training (ImageDataGenerator):
        rescale = 1./255  →  pixel / 255.0
        target_size = (224, 224)

    Pipeline inferensi (app.py) harus IDENTIK:
        1. Buka gambar → konversi ke RGB  (hindari RGBA/grayscale)
        2. Resize ke (224, 224) dengan LANCZOS  (kualitas terbaik)
        3. Konversi ke float32 lalu / 255.0
        4. Tambah dimensi batch → shape (1, 224, 224, 3)

    ✅ FIX 3: Tambah dtype=np.float32 eksplisit dan Image.LANCZOS
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SHAPE, Image.LANCZOS)           # kualitas resize lebih baik
    img_arr = np.array(img, dtype=np.float32) / 255.0     # ← float32 eksplisit
    img_arr = np.expand_dims(img_arr, axis=0)              # (1, H, W, 3)
    return tf.convert_to_tensor(img_arr, dtype=tf.float32)


def decode_base64_image(b64_string: str) -> bytes:
    """
    Decode string Base64 ke bytes.
    Mendukung format dengan header:  "data:image/jpeg;base64,/9j/4AAQ..."
    Atau tanpa header               :  "/9j/4AAQ..."
    """
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,")[1]
    try:
        return base64.b64decode(b64_string)
    except Exception as exc:
        raise ValueError(f"Base64 decoding error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT /predict
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict

    Request body (JSON):
    {
        "file": "data:image/jpeg;base64,/9j/4AAQ..."
    }

    Response sukses (200):
    {
        "label"     : "mangga",
        "confidence": 0.9341,
        "all_probs" : {"apel": 0.02, "mangga": 0.93, ...}
    }

    Response gagal (400):
    {
        "error": "Buah tidak dikenali (confidence terlalu rendah: 0.1234)"
    }
    """

    # 1. Validasi Content-Type
    if not request.is_json:
        return jsonify({"error": "Content-Type harus application/json"}), 400

    data = request.get_json()
    if "file" not in data:
        return jsonify({"error": "Field 'file' tidak ditemukan di JSON body"}), 400

    try:
        # ── Step 1: Decode Base64 → raw bytes ────────────────────────────────
        image_bytes = decode_base64_image(data["file"])

        # ── Step 2: Preprocessing → Tensor ───────────────────────────────────
        img_tensor = preprocess_image_from_bytes(image_bytes)

        # ── Step 3: Prediksi ──────────────────────────────────────────────────
        preds = model.predict(img_tensor, verbose=0)   # shape: (1, num_classes)

        # ✅ FIX 4 — KUNCI UTAMA: JANGAN apply softmax lagi!
        #
        #   Model sudah pakai activation='softmax' di layer Dense terakhir
        #   (lihat training script Cell 8):
        #       outputs = layers.Dense(num_classes, activation='softmax')(x)
        #
        #   Jika kamu apply tf.nn.softmax() lagi di sini pada output yang
        #   sudah softmax, hasilnya adalah distribusi yang sangat flat
        #   (semua kelas mendapat probabilitas hampir sama ~1/num_classes).
        #
        #   Contoh dengan 9 kelas:
        #     Output softmax model → [0.85, 0.05, 0.03, ...]  ← benar
        #     Setelah double softmax → [0.12, 0.11, 0.10, ...]  ← flat/salah
        #
        #   ❌ SALAH (kode lama):
        #       probs = tf.nn.softmax(preds, axis=1).numpy()[0]
        #
        #   ✅ BENAR: langsung ambil output model
        probs = preds[0]   # shape: (num_classes,) — sudah probabilitas 0..1

        # ── Step 4: Ambil prediksi teratas ───────────────────────────────────
        top_idx    = int(np.argmax(probs))
        label      = class_names[top_idx]
        confidence = float(probs[top_idx])

        # ── Step 5: Debug log semua probabilitas ─────────────────────────────
        all_probs = {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))}
        print(f"\n[PREDICT] label={label} | confidence={confidence:.4f}")
        print(f"[PREDICT] all_probs={json.dumps(all_probs, ensure_ascii=False)}")

        # ── Step 6: Threshold ─────────────────────────────────────────────────
        # ✅ FIX 5: Threshold 0.5 lebih rasional daripada 0.12
        #   - 0.12 terlalu ketat → selalu reject karena double softmax menyebabkan
        #     semua confidence ~0.11
        #   - Setelah fix double softmax, confidence buah yang dikenali akan ≥ 0.7-0.9
        #   - Threshold 0.5 = model minimal 50% yakin = prediksi yang cukup kuat
        CONFIDENCE_THRESHOLD = 0.8
        if confidence < CONFIDENCE_THRESHOLD:
             return jsonify({"error": "Buah tidak dikenali"}), 400

        # ── Step 7: Response sukses ───────────────────────────────────────────
        return jsonify({
            "label"     : label,
            "confidence": round(confidence, 4),
            # "all_probs" : all_probs    # opsional, berguna untuk debugging
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()   # print stack trace di terminal server
        return jsonify({"error": str(e)}), 500


# ─── Health Check Endpoint ────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Cek apakah server & model berjalan dengan baik."""
    return jsonify({
        "status" : "ok",
        "model"  : MODEL_PATH,
        "classes": class_names,
        "n_class": len(class_names),
    }), 200


# ─── Jalankan Server ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # debug=True untuk development (auto-reload), False untuk produksi
    app.run(host="0.0.0.0", port=port, debug=False)