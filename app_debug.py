
print("Starting Flask app...")

from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import random

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import register_keras_serializable

# ✅ Fix for quantization_config issue
@register_keras_serializable()
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

app = Flask(__name__)

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load model safely
model = None
try:
    model = load_model(
        "final_model.h5",
        compile=False,
        custom_objects={"Dense": CustomDense}
    )
    print(" Model loaded successfully!")
    print(f" DEBUG: Model type: {type(model)}")
    print(f" DEBUG: Model is None: {model is None}")
    print(f" DEBUG: Model input shape: {model.input_shape}")
    print(f" DEBUG: Model output shape: {model.output_shape}")
except Exception as e:
    print(" Model loading failed:", e)
    print(" Running in DEMO mode")
    print(f" DEBUG: Model is None after loading attempt: {model is None}")

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
# 🔮 Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index.html", prediction="⚠️ No image uploaded")

        # Save uploaded file
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return render_template("index.html", prediction="❌ Invalid image")

        # Preprocess
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.reshape(img, (1, 128, 128, 3))

        # Prediction with debugging
        print(f" DEBUG: Model is None: {model is None}")
        print(f" DEBUG: Model type: {type(model)}")
        print(f" DEBUG: Image shape: {img.shape}")
        print(f" DEBUG: Image min/max: {img.min():.3f} / {img.max():.3f}")
        
        if model is not None:
            print(" DEBUG: Using REAL model for prediction")
            prediction = model.predict(img, verbose=0)[0][0]
            demo_mode = False
            print(f" DEBUG: Raw prediction: {prediction:.6f}")
        else:
            print(" DEBUG: Using RANDOM prediction (DEMO MODE)")
            prediction = random.random()
            demo_mode = True
            print(f" DEBUG: Random prediction: {prediction:.6f}")

        confidence = round(prediction * 100, 2)
        print(f" DEBUG: Confidence: {confidence}%")
        print(f" DEBUG: Demo mode: {demo_mode}")

        if prediction > 0.5:
            result = f"Cancer Detected ({confidence}%)"
        else:
            result = f"No Cancer ({100 - confidence}%)"

        if demo_mode:
            result += " [DEMO MODE]"

        return render_template("index.html", prediction=result, image=filepath)

    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error: {str(e)}")

# 🚀 Run app
if __name__ == "__main__":
    app.run()