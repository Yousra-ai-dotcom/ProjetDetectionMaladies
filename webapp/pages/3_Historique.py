import streamlit as st
import pandas as pd
import os

def show_history():
    st.title("📊 Historique des analyses")

    if not os.path.exists("uploads/history.csv"):
        st.info("Aucune analyse enregistrée pour le moment.")
        return

    history = pd.read_csv("uploads/history.csv", names=["Date", "Utilisateur", "Fichier", "Résultat"])
    user_history = history[history["Utilisateur"] == st.session_state.user]

    if user_history.empty:
        st.info("Aucun historique disponible pour vous.")
    else:
        st.dataframe(user_history[::-1], use_container_width=True)
