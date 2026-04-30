#!/usr/bin/env python3
"""
Test Flask upload endpoint
"""
import requests
import os

print("🧪 Testing Flask upload endpoint...")

# Check if Flask is running
try:
    response = requests.get("http://127.0.0.1:5000")
    if response.status_code == 200:
        print("✅ Flask app is running")
    else:
        print(f"❌ Flask app returned status: {response.status_code}")
        exit(1)
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to Flask app - make sure it's running on http://127.0.0.1:5000")
    exit(1)

# Upload test image
if os.path.exists("static/test_image.jpg"):
    with open("static/test_image.jpg", "rb") as f:
        files = {"image": f}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)
        
    print(f"📤 Upload status: {response.status_code}")
    print(f"📤 Response text: {response.text[:200]}...")
    
    if response.status_code == 200:
        print("✅ Upload test successful - check Flask console for debug output")
    else:
        print("❌ Upload test failed")
else:
    print("❌ Test image not found - run test_prediction.py first")
