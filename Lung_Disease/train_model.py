# train_model.py
# Simple transfer-learning training for Pneumonia vs Normal chest X-rays.
# Expects dataset structure:
# dataset/
#   train/
#     PNEUMONIA/
#     NORMAL/
#   val/
#     PNEUMONIA/
#     NORMAL/

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8  # reduce for quick runs; increase for better results
DATA_DIR = "dataset"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(input_shape=(224,224,3), n_classes=2):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False  # freeze base
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_flow = val_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    model = build_model(input_shape=(*IMG_SIZE,3), n_classes=2)
    model.summary()

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, "lung_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr]
    )

    # Optionally unfreeze some layers and fine-tune
    base = model.layers[0]
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=3,
        callbacks=[checkpoint, reduce_lr]
    )

if __name__ == "__main__":
    main()
