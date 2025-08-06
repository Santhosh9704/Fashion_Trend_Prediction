import tensorflow as tf
import numpy as np
import cv2

# Load your model
model = tf.keras.models.load_model("style_classifier_model.h5")
print("✅ Model loaded successfully!")

# Label names (update if needed)
class_names = ["retro", "streetwear", "monochrome"]

# Load and preprocess a test image
img_path = "test_image.jpg"  # ✅ Replace this with your image file
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Image not found at path: {img_path}")

img = cv2.resize(img, (224, 224)) / 255.0
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"🧠 Predicted Style: {predicted_class.upper()} (Confidence: {confidence:.2%})")
