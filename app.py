# app.py
import os
import sqlite3
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
from PIL import Image

# Config
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DB_PATH = "database.db"
MODEL_PATH = "face_emotionModel.h5"
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Emotion labels from model (standard FER2013 ordering)
EMOTIONS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

app = Flask(__name__, static_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded from", MODEL_PATH)

# DB init
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_emotion TEXT,
            confidence REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_emotion_from_image(image_path):
    # read image (color), detect face, preprocess to 48x48 grayscale, predict
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48,48))
    if len(faces) == 0:
        # if no face found, resize whole image
        face = cv2.resize(gray, (48,48))
    else:
        # pick largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
    face = face.astype('float32') / 255.0
    # model may expect (48,48,1) or (48,48,3). Adjust if needed:
    inp = np.expand_dims(face, axis=0)            # (1,48,48)
    inp = np.expand_dims(inp, -1)                # (1,48,48,1)
    preds = model.predict(inp)[0]
    top_idx = int(np.argmax(preds))
    emotion = EMOTIONS[top_idx]
    confidence = float(preds[top_idx])
    return emotion, confidence

def friendly_message(emotion):
    # Map emotion -> friendly sentence with emoji
    mapping = {
        'happy': "You look happy 😂",
        'surprised': "Wow — you look surprised 😲",
        'sad': "You look a bit sad — take care 💙",
        'angry': "You look angry — breathe deeply 😤",
        'fearful': "You look fearful — everything okay? 😰",
        'disgusted': "You look disgusted 🤢",
        'neutral': "You're looking neutral 🙂"
    }
    return mapping.get(emotion, f"You look {emotion}")

@app.route("/")
def index():
    # serve the HTML file (we assume index.html in project root)
    return send_from_directory('.', 'index.html')

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return jsonify({"error":"No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error":"No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(datetime.utcnow().strftime("%Y%m%d%H%M%S_") + file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        try:
            emotion, confidence = predict_emotion_from_image(path)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        # store metadata
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO uploads (filename, predicted_emotion, confidence, created_at) VALUES (?, ?, ?, ?)",
                  (filename, emotion, confidence, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        return jsonify({
            "message": friendly_message(emotion),
            "emotion": emotion,
            "confidence": confidence,
            "file": url_for('uploaded_file', filename=filename)
        })
    else:
        return jsonify({"error":"Invalid file type"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    # For production use gunicorn. This is for local debugging:
    app.run(host='0.0.0.0', port=5000, debug=True)
