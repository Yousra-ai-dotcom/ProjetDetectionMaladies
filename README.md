# 🌿 AgriScan Pro - Plant Disease Detection

AgriScan Pro is an intelligent web application for detecting plant diseases from a sheet image, combining artificial intelligence and computer vision.

---

## 🎯 Main objective

Allow a user (farmer or technician) to:

- 📷 Upload or capture a plant leaf,

- 🧠 Automatically identify if the plant is healthy or sick,

- 🔍 Detect the specific disease (e.g. mildew, alterniasis, rust...),

- 📋 Get treatment and prevention advice.

---

## ⚙️ Technologies used

- Python (main language)

- Flask: web framework

- TensorFlow + Keras: training of the AI model (MobileNetV2)

- OpenCV: image processing and real-time detection via webcam

- SQLite: user database and history

- HTML / CSS / Bootstrap: stylish user interface

---

## 🧠 Model IA

- MobileNetV2 pre-trained with Transfer Learning

- Trained on the PlantVillage dataset

- Overall accuracy: +95%

- Visualisations included:

- Confusion matrix (`confusion_matrix.png`)

- Learning curves (`training_history.png`)

---

## 🖥️ Main features

- ✅ Image analysis (upload or webcam)

- ✅ Detailed results: name of the disease in French, confidence, symptoms, treatment, prevention

- ✅ Personalised history by user

- ✅ Secure authentication (connection/registration)

- ✅ Real-time detection with dynamic overlay

---

## 📦 Project structure

- `data/`: raw and pre-processed images

- `scripts/`: pre-processing, training, evaluation

- `models/`: trained models and visualisations

- `webapp/`: Flask application (HTML, CSS, Python)

- `auth/`: management of user accounts

- `realtime_detection.py`: live OpenCV prototype

---

## 🚀 Conclusion

AgriScan Pro shows how artificial intelligence can improve agriculture by offering a fast, reliable and accessible tool for the diagnosis of plant diseases.

*This project is open source and intended for educational use. *
