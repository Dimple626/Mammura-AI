print("Starting Mammura AI...")

from flask import Flask, render_template, request
import cv2
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import register_keras_serializable

# Custom Dense layer to handle quantization_config compatibility
@register_keras_serializable()
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

app = Flask(__name__)

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load model
model = None
try:
    model = load_model(
        "final_model.h5",
        compile=False,
        custom_objects={"Dense": CustomDense}
    )
    print("✅ Model loaded successfully!")
    print(f"📋 Model input shape: {model.input_shape}")
    print(f"📋 Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("⚠️ Application will run in DEMO MODE")
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index.html", prediction="⚠️ No image uploaded")

        # Save uploaded file
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Read and validate image
        img = cv2.imread(filepath)
        if img is None:
            return render_template("index.html", prediction="❌ Invalid image")

        # Preprocess image (exact match for training)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.reshape(img, (1, 128, 128, 3))

        # Make prediction
        if model is not None:
            prediction = model.predict(img, verbose=0)[0][0]
            demo_mode = False
        else:
            import random
            prediction = random.random()
            demo_mode = True

        # Calculate confidence and format result
        confidence = round(prediction * 100, 2)
        
        if prediction > 0.5:
            result = f"Cancer Detected ❌ ({confidence}%)"
        else:
            result = f"No Cancer ✅ ({100 - confidence}%)"

        if demo_mode:
            result += " [DEMO MODE]"

        return render_template("index.html", prediction=result, image=filepath)

    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
