import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import datetime

MODEL_PATH = "models/plant_disease_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0

def save_history(user, filename, prediction):
    with open("uploads/history.csv", "a") as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        f.write(f"{now},{user},{filename},{prediction}\n")

def show_analysis():
    st.title("🌿 Analyse d'image")
    model = load_model()

    uploaded_file = st.file_uploader("Choisissez une image de feuille", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image sélectionnée", use_column_width=True)

        img = preprocess_image(image)
        prediction = model.predict(img)

        classes = list(model.output_names) if hasattr(model, 'output_names') else os.listdir("data/processed/train")
        predicted_class = classes[np.argmax(prediction)]

        if "healthy" in predicted_class.lower():
            st.success(f"✅ La plante est saine")
        else:
            st.warning(f"🚨 La plante semble malade : **{predicted_class}**")

        # Historique (optionnel)
        if st.session_state.user:
            save_history(st.session_state.user, uploaded_file.name, predicted_class)
