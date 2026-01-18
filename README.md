```markdown
# 🌿 Plant Doctor AI: Deep Learning Disease Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.0-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end Computer Vision application that detects plant diseases from leaf images with **95%+ accuracy**. 

Built with a custom **Convolutional Neural Network (CNN)**, optimized with **TensorFlow Lite** for edge deployment, and served via a **Flask** web interface.

---

## 🚀 Features

* **Multi-Class Detection:** Identifies 4 classes: Pepper Bacterial Spot, Potato Early Blight, Tomato Late Blight, and Healthy Tomato.
* **High Accuracy:** Trained on the **PlantVillage dataset** using Data Augmentation and Dropout regularization to prevent overfitting.
* **Real-Time Inference:** Uses `TFLite` quantization for sub-second prediction speeds.
* **User-Friendly UI:** Clean frontend with drag-and-drop upload and confidence visualization (Green/Yellow/Red indicators).
* **Production Ready:** Docker-friendly and optimized for cloud deployment (Render/Heroku).

---

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow, Keras, Data Augmentation.
* **Backend:** Flask (Python), Gunicorn.
* **Frontend:** HTML5, TailwindCSS, JavaScript.
* **Image Processing:** Pillow (PIL), NumPy.
* **DevOps:** Render (Cloud Deployment).

---

## 📂 Project Structure

```bash
plant-doctor-ai/
├── model.tflite          # Optimized TFLite model (Quantized)
├── app.py                # Flask backend application
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Frontend UI
├── Plant_Disease_NB.ipynb # (Optional) Jupyter Notebook used for training
└── README.md             # Project Documentation

```

---

## 📊 Model Performance

The model was trained for **50 epochs** with the following strategies to ensure robustness:

1. **Data Augmentation:** Random flips, rotations (20%), and zooms (20%).
2. **Regularization:** `Dropout(0.5)` layer added before the classifier to eliminate overfitting.
3. **Optimization:** Adam optimizer with categorical crossentropy loss.

### Training Metrics

| Metric | Training | Validation |
| --- | --- | --- |
| **Accuracy** | 98.2% | 96.5% |
| **Loss** | 0.05 | 0.12 |

<img width="1156" height="547" alt="download (1)" src="https://github.com/user-attachments/assets/a22fbe6a-3edc-431c-9c01-eb23b9f78e42" />


---

## ⚡ Getting Started

### Prerequisites

* Python 3.8+
* pip (Python Package Manager)

### Installation

1. **Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/Plant-Disease-Detection-CNN.git](https://github.com/YOUR_USERNAME/Plant-Disease-Detection-CNN.git)
cd Plant-Disease-Detection-CNN

```


2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Run the Application**
```bash
python app.py

```


4. **Access the UI**
Open your browser and navigate to: `http://127.0.0.1:5000`


---

## ⚠️ Limitations

* **Background Noise:** The model performs best on images with simple/neutral backgrounds. Complex backgrounds (hands, soil) may reduce confidence.
* **Lighting:** Extreme shadows or overexposure can lead to false positives.
* **Scope:** Currently limited to the 4 classes trained on the PlantVillage subset.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

### 👤 Contact

**Yoosuf** [LinkedIn Profile](https://www.linkedin.com/in/yoosuf-ahamed-21982026a/) | [GitHub Profile](https://github.com/YOOSUF-HUB)
