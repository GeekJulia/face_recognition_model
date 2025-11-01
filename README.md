# face_recognition_model
This repository contains a Face recognition model built with Python and TensorFlow, the database is FER2013
# Face Emotion Detector (Flask + Keras pre-trained model)

Requirements:
- Python 3.8–3.10 recommended
- See requirements.txt

1. Install packages
   pip install -r requirements.txt

2. Download model (creates face_emotionModel.h5)
   python model_training.py

3. Run locally
   python app.py
   Open http://localhost:5000

4. Deploy to Render
   - Push repository to GitHub
   - On Render: New -> Web Service
     - Connect GitHub repo
     - Build Command: pip install -r requirements.txt
     - Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
     - Make sure face_emotionModel.h5 is in the repo (or use a startup script to download via model_training.py)
