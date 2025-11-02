# model_fixer.py

import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

MODEL_PATH = "face_emotionModel.h5"
NEW_MODEL_PATH = "face_emotionModel_fixed.h5"

# The original model used a parameter ('batch_shape') that is now obsolete.
# By passing an empty dictionary as 'custom_objects', we force Keras to try
# to load the model's structure while ignoring or implicitly handling minor 
# serialization errors related to obsolete arguments.

print("Attempting to load model for fixing...")
try:
    # Use the load_model function with compile=False and empty custom_objects
    # This combination often allows Keras to load the structure despite minor config errors.
    model = load_model(
        MODEL_PATH,
        custom_objects={}, 
        compile=False
    )
    print("Model loaded successfully.")

    # Re-save the model using the modern Keras format.
    # This strips out the problematic 'batch_shape' argument and uses the current
    # serialization standard that TF 2.14.0 expects.
    save_model(model, NEW_MODEL_PATH, save_format='h5')
    print(f"Model successfully re-saved to {NEW_MODEL_PATH}")

except Exception as e:
    print(f"Failed to fix model. Error: {e}")