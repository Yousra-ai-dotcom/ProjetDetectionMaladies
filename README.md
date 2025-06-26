# 🌿 AgriScan Pro – Détection des Maladies des Plantes

AgriScan Pro est une application web intelligente de détection des maladies des plantes à partir d’une image de feuille, combinant intelligence artificielle et vision par ordinateur.

---

## 🎯 Objectif principal

Permettre à un utilisateur (agriculteur ou technicien) de :

- 📷 Téléverser ou capturer une feuille de plante,
- 🧠 Identifier automatiquement si la plante est saine ou malade,
- 🔍 Détecter la maladie précise (ex. : mildiou, alternariose, rouille…),
- 📋 Obtenir des conseils de traitement et de prévention.

---

## ⚙️ Technologies utilisées

- Python (langage principal)  
- Flask : framework web  
- TensorFlow + Keras : entraînement du modèle IA (MobileNetV2)  
- OpenCV : traitement d’image et détection temps réel via webcam  
- SQLite : base de données utilisateurs et historique  
- HTML / CSS / Bootstrap : interface utilisateur stylée

---

## 🧠 Modèle IA

- MobileNetV2 pré-entraîné avec Transfer Learning  
- Entraîné sur le dataset PlantVillage  
- Précision globale : +95%  
- Visualisations incluses :  
  - Matrice de confusion (`confusion_matrix.png`)  
  - Courbes d’apprentissage (`training_history.png`)  

---

## 🖥️ Fonctionnalités principales

- ✅ Analyse d’image (upload ou webcam)  
- ✅ Résultats détaillés : nom de la maladie en français, confiance, symptômes, traitement, prévention  
- ✅ Historique personnalisé par utilisateur  
- ✅ Authentification sécurisée (connexion/inscription)  
- ✅ Détection temps réel avec overlay dynamique  

---

## 📦 Structure du projet

- `data/` : images brutes et prétraitées  
- `scripts/` : prétraitement, entraînement, évaluation  
- `models/` : modèles entraînés et visualisations  
- `webapp/` : application Flask (HTML, CSS, Python)  
- `auth/` : gestion des comptes utilisateurs  
- `realtime_detection.py` : prototype OpenCV en direct  

---

## 🚀 Conclusion

AgriScan Pro montre comment l’intelligence artificielle peut améliorer l’agriculture en proposant un outil rapide, fiable et accessible pour le diagnostic des maladies des plantes.

Projet modulaire, complet et évolutif — idéal pour extension mobile ou usage terrain.

---

## 👩‍💻 Auteur

**Yousra Ameur**  
Étudiante en Intelligence Artificielle  
📧 ameuryoussra4@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/yousra-ameur-329839277/)

---

*Ce projet est open source et destiné à un usage pédagogique.*