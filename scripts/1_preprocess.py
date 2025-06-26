import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 📌 Fonction de prétraitement d'une image (lecture, conversion, redimensionnement)
def preprocess_with_opencv(src_path):
    try:
        img = cv2.imread(src_path)  # Lecture de l'image
        if img is None:
            return None  # Retourne None si l'image est illisible
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertit de BGR à RGB
        img = cv2.resize(img, (128, 128))  # Redimensionne à 128x128 pixels
        return img  # Retourne l'image traitée
    except Exception as e:
        print(f"Erreur avec l'image {src_path} : {e}")
        return None  # Retourne None en cas d'erreur

# 📁 Fonction pour séparer les images en train / val / test
def split_data(source_dir="data/raw", test_size=0.2):
    # Liste des sous-dossiers (chaque sous-dossier représente une classe)
    classes = [c for c in os.listdir(source_dir) 
               if not c.startswith('.') and os.path.isdir(os.path.join(source_dir, c))]

    if not classes:
        raise ValueError(f"Aucune classe trouvée dans {source_dir}")

    # Parcours de chaque classe
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        
        # Récupère les noms des fichiers image valides
        images = [f for f in os.listdir(cls_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]

        if not images:
            print(f"⚠ Aucune image trouvée dans {cls_path}")
            continue  # Passe à la classe suivante s’il n’y a pas d’images

        # ✂️ Séparation en train (80%) + temp (20%)
        train, temp = train_test_split(images, test_size=test_size, random_state=42)
        # ✂️ Séparation de temp en val (10%) et test (10%)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        # Création des dossiers de sortie s'ils n'existent pas
        for subset_name in ['train', 'val', 'test']:
            os.makedirs(os.path.join("data/processed", subset_name, cls), exist_ok=True)

        # Traitement et copie des images dans les bons dossiers
        for subset, subset_name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            for img_name in subset:
                src = os.path.join(cls_path, img_name)
                dst = os.path.join("data/processed", subset_name, cls, img_name)

                # Appliquer le prétraitement
                img = preprocess_with_opencv(src)
                if img is not None:
                    # Sauvegarde l’image traitée
                    cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                else:
                    print(f"⚠ Image ignorée (non lisible) : {src}")

# ▶️ Point d’entrée du script
if __name__ == "__main__":
    print("🔍 Vérification des données...")
    try:
        split_data()
        print("✅ Prétraitement terminé avec succès !")
    except Exception as e:
        print(f"❌ Erreur : {e}")
        print("1. Vérifie que le dataset est dans data/raw/")
        print("2. Les sous-dossiers doivent représenter les classes")
        print("3. Formats supportés : .jpg / .png")