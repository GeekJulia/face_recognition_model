# model_training.py
# Downloads a pre-trained Keras model from Hugging Face and saves it as face_emotionModel.h5
from huggingface_hub import hf_hub_download
from tensorflow import keras
import shutil
import os

MODEL_REPO = "shivamprasad1001/Emo0.1"   # model page: Emo0.1 (FER2013)
MODEL_FILENAME = "Emo0.1.h5"
OUTNAME = "face_emotionModel.h5"

def download_and_save():
    print("Downloading model from Hugging Face...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    print("Downloaded to:", model_path)
    # Copy/rename to project name
    shutil.copyfile(model_path, OUTNAME)
    print(f"Saved model as {OUTNAME}")

if __name__ == "__main__":
    download_and_save()
