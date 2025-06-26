import streamlit as st

def show_home():
    st.markdown("""
        <div style='text-align: center;'>
            <img src='webapp/static/images/logo.png' width='150'>
            <h1 style='color: #2E7D32;'>Bienvenue sur AgriScan Pro 🌿</h1>
            <h3 style='color: #555;'>Détection intelligente des maladies des plantes</h3>
        </div>
    """, unsafe_allow_html=True)

    st.write("""
    **AgriScan Pro** est une application intelligente de vision par ordinateur pour les agriculteurs, étudiants et chercheurs.
    Elle permet de :

    - ✅ Classifier une plante comme *saine* ou *malade*
    - 🔬 Identifier le type de maladie
    - 🧠 Utiliser un modèle entraîné avec MobileNetV2
    - 📊 Consulter l'historique des analyses
    """)

    st.info("Commencez dès maintenant via le menu à gauche 🌿")

