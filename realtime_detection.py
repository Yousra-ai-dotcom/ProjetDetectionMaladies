import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle entraîné
model = tf.keras.models.load_model('models/plant_disease_model.h5')

# Liste des classes (à adapter selon ton dataset)
import os
CLASS_NAMES = sorted(os.listdir("data/processed/train"))

DISPLAY_NAMES = {
    "Apple__Apple_scab": "Pomme : Tavelure",
    "Apple__Black_rot": "Pomme : Pourriture noire",
    "Apple__Cedar_apple_rust": "Pomme : Rouille grillagée",
    "Apple__healthy": "Pomme saine",

    "Corn__Cercospora_leaf_spot": "Maïs : Tache foliaire Cercospora",
    "Corn__Common_rust": "Maïs : Rouille commune",
    "Corn__Northern_Leaf_Blight": "Maïs : Brûlure des feuilles du nord",
    "Corn__healthy": "Maïs sain",

    "Grape__Black_rot": "Vigne : Pourriture noire",
    "Grape__Esca": "Vigne : Esca",
    "Grape__Leaf_blight": "Vigne : Brûlure des feuilles",
    "Grape__healthy": "Vigne saine",

    "Pepper__bell__Bacterial_spot": "Poivron : Tache bactérienne",
    "Pepper__bell__healthy": "Poivron sain",

    "Potato___Early_blight": "Pomme de terre : Alternariose",
    "Potato___Late_blight": "Pomme de terre : Mildiou",
    "Potato___healthy": "Pomme de terre saine",

    "Tomato__Bacterial_spot": "Tomate : Tache bactérienne",
    "Tomato__Early_blight": "Tomate : Alternariose",
    "Tomato__Late_blight": "Tomate : Mildiou",
    "Tomato__Leaf_Mold": "Tomate : Moisissure des feuilles",
    "Tomato__Septoria_leaf_spot": "Tomate : Tache septorienne",
    "Tomato__Spider_mites_Two_spotted_spider_mite": "Tomate : Araignées rouges",
    "Tomato__Target_Spot": "Tomate : Tache cible",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomate : Virus de l’enroulement jaune",
    "Tomato__Tomato_mosaic_virus": "Tomate : Virus de la mosaïque",
    "Tomato__healthy": "Tomate saine"
}

# Initialiser webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture webcam")
        break

    # Prétraiter l'image capturée
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Prédiction
    preds = model.predict(img)[0]
    idx = np.argmax(preds)
    label = DISPLAY_NAMES.get(CLASS_NAMES[idx], CLASS_NAMES[idx])
    confidence = preds[idx]

    # Texte à afficher
    text = f"{label} ({confidence*100:.1f}%)"

    # Choix couleur selon santé/plante saine
    if 'healthy' in CLASS_NAMES[idx].lower():
        color = (0, 255, 0)  # Vert
    else:
        color = (0, 0, 255)  # Rouge

    # Affichage du texte sur la vidéo
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Affichage de la vidéo avec overlay
    cv2.imshow("AgriScan Pro - Détection en temps réel", frame)

    # Sortie sur touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()