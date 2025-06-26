#Flask pour créer une application web avec gestion des routes, des sessions, des redirections et des fichiers statiques.
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
#gérer les chemins et insérer dynamiquement des modules dans le chemin Python.
import sys
import os
#Ajoute le répertoire parent au sys.path pour importer des modules personnalisés comme auth.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#Modules essentiels pour le traitement d’image (OpenCV, NumPy, PIL), base de données (SQLite), upload sécurisé (Werkzeug), deep learning (TensorFlow).
import cv2
import numpy as np
import sqlite3
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from auth.login import authenticate
from auth.register import register_user
#Dictionnaire contenant les maladies par type de plante, avec nom affiché, symptômes, traitements, prévention, gravité et couleur associée.
DISEASE_DATABASE = {
    # ============ POMMES ============
    "Apple__Apple_scab": {
        "display_name": "Pomme : Tavelure",
        "symptoms": "Taches brun-olive sur feuilles, fruits tachés ou déformés",
        "treatment": [
            "Pulvérisations de fongicides (captane, dodine) pendant la période de croissance",
            "Élimination des feuilles et fruits tombés"
        ],
        "prevention": [
            "Utilisation de variétés résistantes (Florina, Prima)",
            "Taille pour améliorer la circulation de l'air"
        ],
        "severity": "Modérée à élevée",
        "color": "#556B2F"  # Vert kaki foncé
    },
    "Apple__Black_rot": {
        "display_name": "Pomme : Pourriture noire",
        "symptoms": "Lésions brun foncé sur feuilles et fruits, fruits momifiés",
        "treatment": [
            "Application de fongicides (Captane, Soufre)",
            "Élimination des fruits infectés",
            "Taille des branches malades"
        ],
        "prevention": [
            "Bonne circulation d'air entre les arbres",
            "Nettoyage des feuilles mortes à l'automne",
            "Pulvérisation préventive au printemps"
        ],
        "severity": "Élevée",
        "color": "#8B0000"  # Rouge foncé
    },
    "Apple__Cedar_apple_rust": {
        "display_name": "Pomme : Rouille grillagée",
        "symptoms": "Taches jaune-orange sur feuilles, excroissances sur fruits",
        "treatment": [
            "Fongicides à base de myclobutanil",
            "Élimination des genévriers voisins (hôtes alternatifs)"
        ],
        "prevention": ["Planter des variétés résistantes (Liberty, Freedom)"],
        "severity": "Modérée",
        "color": "#FFA500"  # Orange
    },

    # ============ TOMATES ============
    "Tomato__Early_blight": {
        "display_name": "Tomate : Alternariose",
        "symptoms": "Taches concentriques brunes avec halos jaunes sur feuilles âgées",
        "treatment": [
            "Fongicides (Chlorothalonil, Mancozèbe)",
            "Suppression des feuilles infectées"
        ],
        "prevention": [
            "Rotation des cultures (3-4 ans)",
            "Paillage pour éviter les éclaboussures",
            "Espacement des plants (60cm)"
        ],
        "severity": "Modérée",
        "color": "#A0522D"  # Brun
    },
    "Tomato__healthy": {
        "display_name": "Tomate : Plante saine",
        "symptoms": "Feuilles vert foncé uniformes, tiges vigoureuses",
        "treatment": ["Aucun traitement nécessaire"],
        "prevention": [
            "Contrôle régulier des parasites",
            "Arrosage au pied sans mouiller le feuillage"
        ],
        "severity": "Aucune",
        "color": "#228B22"  # Vert forêt
    },

    # ============ MAÏS ============
    "Corn__Common_rust": {
        "display_name": "Maïs : Rouille commune",
        "symptoms": "Pustules poudreuses brun-rouille sur les deux faces des feuilles",
        "treatment": [
            "Application de fongicides (Azoxystrobine) si nécessaire",
            "Éviter les excès d'azote"
        ],
        "prevention": ["Utiliser des hybrides résistants (DKC, Pioneer)"],
        "severity": "Faible à modérée",
        "color": "#CD5C5C"  # Rouge indien
    },
    "Corn__Northern_Leaf_Blight": {
        "display_name": "Maïs : Helminthosporiose",
        "symptoms": "Longues lésions grises avec bordures vert olive",
        "treatment": ["Fongicides en début d'infection"],
        "prevention": ["Labour profond pour enfouir les résidus"],
        "severity": "Élevée en conditions humides",
        "color": "#808000"  # Olive
    },

    # ============ POMMES DE TERRE ============
    "Potato___Early_blight": {
        "display_name": "Pomme de terre : Alternariose",
        "symptoms": "Taches angulaires brunes avec anneaux concentriques",
        "treatment": ["Pulvérisations fongicides tous les 7-10 jours"],
        "prevention": ["Éviter le stress hydrique"],
        "severity": "Modérée",
        "color": "#6B8E23"  # Vert olive foncé
    },
    "Potato___Late_blight": {
        "display_name": "Pomme de terre : Mildiou",
        "symptoms": "Taches huileuses devenant noires, moisissure blanche au revers",
        "treatment": ["Fongicides systémiques (Metalaxyl)"],
        "prevention": ["Plants certifiés sans maladie"],
        "severity": "Très élevée",
        "color": "#483D8B"  # Bleu ardoise foncé
    },

    # ============ RAISINS ============
    "Grape__Black_rot": {
        "display_name": "Vigne : Pourriture noire",
        "symptoms": "Taches brunes avec points noirs, baies ratatinées",
        "treatment": ["Fongicides préventifs avant floraison"],
        "prevention": ["Taille pour aérer la végétation"],
        "severity": "Critique",
        "color": "#800080"  # Violet
    },
    "Grape__Leaf_blight": {
        "display_name": "Vigne : Brûlure bactérienne",
        "symptoms": "Nécroses marginales des feuilles",
        "treatment": ["Pulvérisations cupriques"],
        "prevention": ["Désinfection des outils de taille"],
        "severity": "Modérée",
        "color": "#9932CC"  # Violet foncé
    }
}

# Version simplifiée pour l'affichage
DISEASE_DISPLAY_NAMES = {k: v["display_name"] for k, v in DISEASE_DATABASE.items()}

app = Flask(__name__)
app.secret_key = 'une_clef_secrete_pour_la_session'

#Définition du répertoire d’upload pour stocker les images envoyées par les utilisateurs.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload

# Charger modèle TensorFlow (au démarrage)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'plant_disease_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Classe à adapter à ton dataset PlantVillage (ordre alphabétique)
CLASS_NAMES = sorted(os.listdir(os.path.join(BASE_DIR, '..', 'data', 'processed', 'train')))
#Fonction utilitaire pour se connecter à la base de données des utilisateurs et de l’historique.
def get_db_connection():
    conn = sqlite3.connect(os.path.join(BASE_DIR, '..', 'users.db'))
    conn.row_factory = sqlite3.Row
    return conn
#Prétraitement des images Utilise OpenCV 
def preprocess_image_cv(img_path):
    # Lire l'image en couleur
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image {img_path}")

    # Convertir BGR (OpenCV) en RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Optionnel : appliquer un flou pour réduire le bruit
    img = cv2.GaussianBlur(img, (3,3), 0)

    # Redimensionner à la taille attendue par le modèle
    img = cv2.resize(img, (128, 128))

    # Normaliser les pixels entre 0 et 1
    img = img.astype('float32') / 255.0

    # Ajouter la dimension batch
    img = np.expand_dims(img, axis=0)

    return img

#Redirige vers la page de connexion si l’utilisateur n’est pas connecté.
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

# --- Auth ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        if authenticate(conn, username, password):
            session['username'] = username
            flash('Connexion réussie', 'success')
            return redirect(url_for('home'))
        else:
            flash('Nom d\'utilisateur ou mot de passe incorrect', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm']
        if password != confirm:
            flash('Les mots de passe ne correspondent pas', 'warning')
        else:
            conn = get_db_connection()
            if register_user(conn, username, password):
                flash('Inscription réussie, connectez-vous', 'success')
                return redirect(url_for('login'))
            else:
                flash('Nom d\'utilisateur déjà pris', 'danger')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Déconnexion réussie', 'info')
    return redirect(url_for('login'))

# --- Pages principales ---

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])
@app.route('/realtime')
def realtime():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('realtime.html')


@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Initialisation des variables pour éviter l'erreur
    result = None
    confidence = None
    filename = None
    health_status = None
    symptoms = None
    treatment = None
    prevention = None
    result_display = None  # <-- initialisation ici !

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('Aucun fichier sélectionné', 'warning')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('Aucun fichier sélectionné', 'warning')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = preprocess_image_cv(filepath)
            preds = model.predict(img)[0]
            top_idx = np.argmax(preds)
            result = CLASS_NAMES[top_idx]
            confidence = preds[top_idx]

            disease_info = DISEASE_DATABASE.get(result)
            if disease_info:
                result_display = disease_info["display_name"]
                symptoms = disease_info.get("symptoms", "Information non disponible")
                treatment = disease_info.get("treatment", ["Information non disponible"])
                prevention = disease_info.get("prevention", ["Information non disponible"])
            else:
                result_display = result
                symptoms = "Information non disponible"
                treatment = ["Information non disponible"]
                prevention = ["Information non disponible"]

            if 'healthy' in result.lower():
                health_status = "Plante saine"
            else:
                health_status = "Maladie détectée : " + result_display

            # Enregistrer historique
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO history (username, image_path, health_status, disease_detected, confidence) VALUES (?, ?, ?, ?, ?)',
                (session['username'], filename,
                 health_status,
                 result,
                 float(confidence))
            )
            conn.commit()
            conn.close()

    return render_template('analyse.html',
                           result=result_display,
                           health_status=health_status,
                           confidence=confidence,
                           filename=filename,
                           symptoms=symptoms,
                           treatment=treatment,
                           prevention=prevention)
                           
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/historique')
def historique():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT * FROM history WHERE username = ? ORDER BY analysis_date DESC',
        (session['username'],)
    ).fetchall()

    rows = [dict(row) for row in rows]
    for row in rows:
        disease_code = row['disease_detected']
        row['disease_display'] = DISEASE_DISPLAY_NAMES.get(disease_code, disease_code)

        if 'healthy' in disease_code.lower():
            row['health_display'] = "Plante saine"
        else:
            row['health_display'] = "Malade"
    conn.close()
    return render_template('historique.html', rows=rows)

@app.route('/apropos')
def apropos():
    return render_template('apropos.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
from flask import Response
import threading

camera = cv2.VideoCapture(0)  # Webcam

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize et normalise pour prédiction
            small = cv2.resize(frame, (128, 128))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            norm = rgb.astype('float32') / 255.0
            input_tensor = np.expand_dims(norm, axis=0)

            # Prédiction
            preds = model.predict(input_tensor)[0]
            idx = np.argmax(preds)
            label = CLASS_NAMES[idx]
            confidence = preds[idx] * 100

            display = DISEASE_DISPLAY_NAMES.get(label, label)

            color = (0, 255, 0) if "healthy" in label.lower() else (0, 0, 255)
            text = f"{display} ({confidence:.1f}%)"

            # Affichage du texte sur l'image
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Encodage pour MJPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)