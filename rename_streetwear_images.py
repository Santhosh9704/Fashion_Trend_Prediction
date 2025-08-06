import os

folder = "fashion_styles/streetwear"
files = os.listdir(folder)
image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()  # Sort alphabetically

for i, filename in enumerate(image_files, start=1):
    ext = os.path.splitext(filename)[1]  # get file extension
    new_name = f"street_{i:02}{ext}"
    src = os.path.join(folder, filename)
    dst = os.path.join(folder, new_name)
    os.rename(src, dst)
    print(f"✅ Renamed: {filename} → {new_name}")

print(f"\n🎉 All {len(image_files)} images renamed successfully!")
