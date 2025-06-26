import streamlit as st

def show_about():
    st.title("ℹ️ À propos d'AgriScan Pro")

    st.markdown("""
    **AgriScan Pro** est un projet développé par Yousra 🌿

    Cette application utilise l'apprentissage profond pour :
    - Détecter les maladies courantes sur les feuilles des plantes
    - Faciliter l'identification précoce de problèmes agricoles
    - Encourager l'usage de l'IA dans l'agriculture durable

    **Technologies utilisées :**
    - Python / TensorFlow / Keras
    - Streamlit pour l'interface
    - MobileNetV2 (Transfer Learning)
    - Dataset PlantVillage

    **Université :** Euro-Mediterranean University of FES
    """)
