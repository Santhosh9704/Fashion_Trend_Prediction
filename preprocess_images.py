import os
import cv2

# Folder containing original images
input_folder = "runway_images"

# Folder to save processed images
output_folder = "processed_images"
os.makedirs(output_folder, exist_ok=True)

image_files = os.listdir(input_folder)
print(f"📸 Found {len(image_files)} images in '{input_folder}' folder")

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    print(f"🔄 Processing: {image_name}")

    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Could not read: {image_name}")
        continue

    resized_img = cv2.resize(img, (224, 224))
    normalized_img = resized_img / 255.0

    save_path = os.path.join(output_folder, image_name)
    cv2.imwrite(save_path, (normalized_img * 255).astype("uint8"))

    print(f"✅ Saved to: {save_path}")
