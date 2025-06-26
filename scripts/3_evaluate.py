import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement du modèle et des données
model = tf.keras.models.load_model('models/quick_model.h5')
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'data/processed/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 2. Prédictions
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)

# 3. Métriques
print("📊 Rapport de Classification :")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# 4. Matrice de confusion
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de Confusion")
plt.savefig('models/confusion_matrix.png')
plt.close()