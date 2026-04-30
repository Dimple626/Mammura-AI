#!/usr/bin/env python3
"""
Test the prediction pipeline with a real image
"""
import numpy as np
import cv2
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import register_keras_serializable

# Custom Dense layer
@register_keras_serializable()
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

print("🧪 Testing prediction pipeline...")

# Load model
try:
    model = load_model("final_model.h5", compile=False, custom_objects={"Dense": CustomDense})
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Create a test image (128x128x3)
test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
print(f"📊 Test image shape: {test_image.shape}")
print(f"📊 Test image dtype: {test_image.dtype}")

# Save test image
cv2.imwrite("static/test_image.jpg", test_image)
print("💾 Saved test image to static/test_image.jpg")

# Preprocess exactly like the Flask app
img = cv2.imread("static/test_image.jpg")
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.reshape(img, (1, 128, 128, 3))

print(f"🔄 Preprocessed image shape: {img.shape}")
print(f"🔄 Preprocessed image min/max: {img.min():.3f} / {img.max():.3f}")
print(f"🔄 Preprocessed image dtype: {img.dtype}")

# Make prediction
prediction = model.predict(img, verbose=0)[0][0]
print(f"🔮 Raw prediction: {prediction:.6f}")
print(f"🔮 Prediction type: {type(prediction)}")

# Interpret result
confidence = round(prediction * 100, 2)
if prediction > 0.5:
    result = f"Cancer Detected ❌ ({confidence}%)"
else:
    result = f"No Cancer ✅ ({100 - confidence}%)"

print(f"🎯 Final result: {result}")
print("✅ Prediction pipeline test complete!")
