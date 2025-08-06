import os

dataset_path = "fashion_styles"
deleted = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        if not os.path.exists(file_path):
            print(f"⚠️ Ghost file reference found (already missing): {file_path}")
            continue
        # If filename is too long or suspicious (like y2jnvo-l... type)
        if len(file) > 180 or file.startswith("._") or file.lower().endswith(".ds_store"):
            try:
                os.remove(file_path)
                print(f"❌ Removed: {file_path}")
                deleted += 1
            except Exception as e:
                print(f"⚠️ Could not delete {file_path}: {e}")

print(f"\n✅ Deep cleaned dataset — {deleted} files removed.\n")
