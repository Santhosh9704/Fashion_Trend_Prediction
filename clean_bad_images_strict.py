from PIL import Image
import os

folder = "fashion_styles"
bad_files = 0

for subdir, _, files in os.walk(folder):
    for file in files:
        file_path = os.path.join(subdir, file)
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            if os.path.exists(file_path):
                print(f"❌ Deleting: {file_path}")
                os.remove(file_path)
                bad_files += 1

print(f"\n✅ Done: {bad_files} bad images removed.")
