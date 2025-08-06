import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- STEP 1: CLEAN CORRUPTED IMAGES ---------------- #
def clean_bad_images(folder):
    removed = 0
    for subdir, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                if os.path.exists(file_path):
                    print(f"❌ Removing corrupted: {file_path}")
                    os.remove(file_path)
                    removed += 1
                else:
                    print(f"⚠️ Skipped missing: {file_path}")
    print(f"\n✅ Cleaned: Removed {removed} bad images.\n")

base_dir = "fashion_styles"
clean_bad_images(base_dir)

# ---------------- STEP 2: DATA LOADING & AUGMENTATION ---------------- #
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# ---------------- STEP 3: BUILD CNN MODEL ---------------- #
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- STEP 4: TRAIN THE MODEL ---------------- #
history = model.fit(
    train_data,
    epochs=2,  # 🔁 You can increase this to 10–15 for better accuracy
    validation_data=val_data
)

# ---------------- STEP 5: SAVE MODEL SAFELY ---------------- #
model.save("style_classifier_model.h5", include_optimizer=False)
# Optional: Save in SavedModel format too
# model.save("style_model")

print("✅ Model saved as style_classifier_model.h5")

# ---------------- STEP 6: PLOT TRAINING CURVE ---------------- #
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()
