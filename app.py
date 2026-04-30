print("Starting Flask app...")

from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import sys

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# ✅ Load FULL model with error handling
model = None
try:
    # Try loading with custom objects to handle compatibility issues
    model = load_model("final_model.h5", compile=False, custom_objects={'quantization_config': None})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    try:
        # Fallback: try loading without custom objects
        model = load_model("final_model.h5", compile=False)
        print("Model loaded successfully with fallback!")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        print("WARNING: Running in demo mode without ML model - will return random predictions")
        print("Please check if the model file is compatible with your TensorFlow version")
        # Don't exit, continue without model for demo purposes


# 🏠 Home route
@app.route('/')
def home():
    return render_template("index.html")


# 🔮 Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']

        if file and file.filename != '':
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            # Read and process image
            img = cv2.imread(filepath)
            if img is None:
                return render_template("index.html", prediction="Error: Could not read image file")

            img = cv2.resize(img, (128,128))
            img = img / 255.0
            img = np.reshape(img, (1,128,128,3))

            # Make prediction
            if model is not None:
                prediction = model.predict(img)[0][0]
                confidence = round(prediction * 100, 2)
                
                if prediction > 0.5:
                    result = f"Cancer Detected ❌ ({confidence}%)"
                else:
                    result = f"No Cancer ✅ ({100 - confidence}%)"
            else:
                # Demo mode: random prediction
                import random
                prediction = random.random()
                confidence = round(prediction * 100, 2)
                
                if prediction > 0.5:
                    result = f"Cancer Detected ❌ ({confidence}%) [DEMO MODE]"
                else:
                    result = f"No Cancer ✅ ({100 - confidence}%) [DEMO MODE]"

            return render_template("index.html", prediction=result, image=filepath)

        return render_template("index.html", prediction="No image uploaded")
    
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


# 🚀 Run app
if __name__ == "__main__":
    app.run(debug=True)
