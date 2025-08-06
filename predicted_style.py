import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("style_classifier_model.h5")

# Label mapping (same order used during training)
class_names = ["retro", "streetwear", "monochrome"]

# Load and preprocess image
img_path = "test_image.jpg"  # Change this to your test image name
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)

# Predict
pred = model.predict(img)
predicted_class = class_names[np.argmax(pred)]

print(f"🧠 Predicted Style: {predicted_class}")
