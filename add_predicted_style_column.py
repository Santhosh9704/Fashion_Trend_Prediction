import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import os

# Load model
model = tf.keras.models.load_model("style_classifier_model.keras")
class_names = ["retro", "streetwear", "monochrome"]

# Load CSV
df = pd.read_csv("caption_sentiment.csv")

# Check if image_path column exists
if "image_path" not in df.columns:
    print("❌ 'image_path' column missing.")
    exit()

predicted_styles = []

# Predict style for each image
for path in df["image_path"]:
    try:
        img = cv2.imread(path)
        if img is None:
            predicted_styles.append("unknown")
            continue
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        predicted_styles.append(predicted_class)
    except Exception as e:
        predicted_styles.append("error")

# Add to DataFrame
df["predicted_style"] = predicted_styles

# Save updated CSV
df.to_csv("caption_sentiment.csv", index=False)
print("✅ 'predicted_style' column added and CSV saved.")