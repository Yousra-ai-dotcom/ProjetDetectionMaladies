import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# 1. Chargement des données (taille réduite pour la vitesse)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/processed/train',
    target_size=(128, 128),  # Taille réduite = gain de vitesse
    batch_size=32,
    class_mode='categorical'
)

# 2. Modèle MobileNetV2 pré-entraîné (Transfer Learning)
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # On ne ré-entraîne pas les poids existants

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# 3. Compilation simplifiée
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Entraînement court (5 epochs suffisent)
history = model.fit(
    train_generator,
    epochs=5,  # Très rapide grâce au Transfer Learning
    verbose=1
)

# 5. Sauvegarde pour déploiement
model.save('models/quick_model.h5')