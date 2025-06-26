import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt

# 1. Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = 'models/plant_disease_model.h5'
os.makedirs('models', exist_ok=True)

# 2. Préparation des données
try:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        'data/processed/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'data/processed/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    print(f"Classes détectées: {train_generator.class_indices}")

except Exception as e:
    raise SystemExit(f"Erreur de chargement des données: {str(e)}")

# 3. Construction du modèle
try:
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )

    print(model.summary())

except Exception as e:
    raise SystemExit(f"Erreur de construction du modèle: {str(e)}")

# 4. Entraînement avec callbacks
try:
    callbacks = [
        EarlyStopping(patience=3, monitor='val_loss'),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Visualisation des résultats
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.savefig('models/training_history.png')
    plt.close()

    print(f"Modèle sauvegardé avec succès à {MODEL_SAVE_PATH}")

except Exception as e:
    raise SystemExit(f"Erreur lors de l'entraînement: {str(e)}")