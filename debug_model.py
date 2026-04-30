#!/usr/bin/env python3
"""
Comprehensive model debugging script
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import register_keras_serializable

print("=" * 60)
print("🔍 MODEL DEBUGGING SCRIPT")
print("=" * 60)

# 1. Check TensorFlow/Keras versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# 2. Check model file
print(f"\n📁 Model file exists: {os.path.exists('final_model.h5')}")
if os.path.exists('final_model.h5'):
    print(f"📊 Model file size: {os.path.getsize('final_model.h5'):,} bytes")

# 3. Custom Dense layer for compatibility
@register_keras_serializable()
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

# 4. Try loading model
print("\n🔄 Attempting to load model...")
model = None
try:
    custom_objects = {"Dense": CustomDense}
    model = load_model("final_model.h5", compile=False, custom_objects=custom_objects)
    print("✅ Model loaded successfully!")
    
    # 5. Model details
    print(f"📋 Model type: {type(model)}")
    print(f"📋 Model is None: {model is None}")
    print(f"📋 Model input shape: {model.input_shape}")
    print(f"📋 Model output shape: {model.output_shape}")
    print(f"📋 Number of layers: {len(model.layers)}")
    
    # 6. Check first few layers
    print("\n🏗️  First 5 layers:")
    for i, layer in enumerate(model.layers[:5]):
        print(f"  {i+1}. {layer.name} ({type(layer).__name__})")
    
    # 7. Check if model has weights
    print(f"\n⚖️  Model has weights: {len(model.weights) > 0}")
    print(f"⚖️  Number of weight arrays: {len(model.weights)}")
    
    # 8. Create test input
    print("\n🧪 Creating test input...")
    test_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
    print(f"📊 Test input shape: {test_input.shape}")
    print(f"📊 Test input min/max: {test_input.min():.3f} / {test_input.max():.3f}")
    
    # 9. Make prediction
    print("\n🔮 Making test prediction...")
    prediction = model.predict(test_input, verbose=0)
    print(f"📊 Prediction shape: {prediction.shape}")
    print(f"📊 Prediction value: {prediction[0][0]:.6f}")
    print(f"📊 Prediction type: {type(prediction[0][0])}")
    
    # 10. Interpret prediction
    pred_value = prediction[0][0]
    confidence = round(pred_value * 100, 2)
    if pred_value > 0.5:
        result = f"Cancer Detected ❌ ({confidence}%)"
    else:
        result = f"No Cancer ✅ ({100 - confidence}%)"
    print(f"🎯 Result: {result}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🏁 DEBUGGING COMPLETE")
print("=" * 60)
