
import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("final_model.h5", compile=False)

def predict_image(image):
    img = cv2.resize(image, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return f"Cancer Detected ❌ ({round(prediction*100,2)}%)"
    else:
        return f"No Cancer ✅ ({round((1-prediction)*100,2)}%)"

# Gradio UI
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="🧠 Mammura AI",
    description="Upload a breast tissue image to detect cancer"
)

interface.launch()