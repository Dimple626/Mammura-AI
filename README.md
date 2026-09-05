# 🎗️ Mammura AI — Breast Cancer Detection System

Mammura AI is an intelligent, deep learning–based web application designed to assist in **early breast cancer detection** using medical imaging. It analyzes histopathology images and classifies them as:

- ❌ **Malignant** (Cancer)
- ✅ **Benign** (Non-Cancer)

Built with TensorFlow/Keras and served through a Flask backend with a modern, interactive web interface.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is **not** a certified medical diagnostic tool and should never be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 📖 Overview

Mammura AI takes a medical image as input, preprocesses it, runs it through a trained convolutional neural network, and returns a prediction along with a confidence score — helping visualize how AI can assist in the early screening of breast cancer.

---

## 📊 Dataset

The model is trained on the **Breast Histopathology Images (IDC) dataset** from Kaggle.

- 📂 **Total Images:** ~277,000+
- **Classes:**
  - `0` → Benign (Non-Cancer)
  - `1` → Malignant (Cancer)
- Images resized to **128x128 pixels** for training

This dataset contains real microscopic breast tissue images used for invasive ductal carcinoma (IDC) detection.

🔗 **Dataset link:** [Breast Histopathology Images (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

---

## ✨ Features

- 🧠 AI-powered image classification (benign vs. malignant)
- 🖼️ Image upload with live preview
- 📈 Confidence-based prediction output
- 💻 Modern and interactive UI
- ⚡ Fast, lightweight Flask backend

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Procfile / `runtime.txt` (Heroku-style) |

---

## ⚙️ How It Works

1. User uploads a medical image through the web interface.
2. The image is preprocessed (resized and normalized to match model input).
3. The trained deep learning model analyzes the image.
4. The app displays the predicted class (Benign/Malignant) along with a confidence score.

---

## 📂 Project Structure

```
Mammura-AI/
├── templates/           # HTML templates for the Flask web app
├── app.py               # Main Flask application
├── final_model.h5        # Trained Keras model
├── model.weights.h5      # Model weights
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python runtime version (for deployment)
├── Procfile              # Process file for deployment (e.g. Heroku)
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x (see `runtime.txt` for the exact version used)
- `pip` for installing dependencies

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Dimple626/Mammura-AI.git
   cd Mammura-AI
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000` (or the port shown in your terminal).

### Usage

1. Open the web app in your browser.
2. Upload a breast histopathology image.
3. Click predict/submit.
4. View the predicted class (Benign/Malignant) and the model's confidence score.

---

## 🌐 Deployment

This project includes a `Procfile` and `runtime.txt`, making it ready for deployment on platforms like **Heroku** or similar PaaS providers:

```bash
git push heroku main
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

No license has been specified for this repository yet.

---

## 🙏 Acknowledgements

- [Breast Histopathology Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) by Paul Mooney on Kaggle
- TensorFlow / Keras community
