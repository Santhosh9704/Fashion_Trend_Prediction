import os
import requests
import math

API_KEY = "UM0YAw5dJAToJWpu75VFCntbYrC8SitMtgDgaTWCICZo1CojngPjxuMW"  # Replace with your actual Pexels API key
HEADERS = {"Authorization": API_KEY}
BASE_URL = "https://api.pexels.com/v1/search"

CATEGORIES = {
    "retro": "retro fashion",
    "streetwear": "streetwear outfit",
    "monochrome": "monochrome fashion"
}
IMAGES_PER_CATEGORY = 100
PER_PAGE = 80  # Pexels max per page is 80

def download_category(cat_name, query):
    folder = os.path.join("fashion_styles", cat_name)
    os.makedirs(folder, exist_ok=True)

    total_pages = math.ceil(IMAGES_PER_CATEGORY / PER_PAGE)
    count = 0

    for page in range(1, total_pages + 1):
        params = {"query": query, "per_page": PER_PAGE, "page": page}
        resp = requests.get(BASE_URL, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        photos = data.get("photos", [])
        for i, photo in enumerate(photos):
            if count >= IMAGES_PER_CATEGORY:
                break
            url = photo["src"]["medium"]  # can choose 'original' if size is desired
            try:
                img_data = requests.get(url).content
                fname = f"{cat_name}_{page}_{i}.jpg"
                with open(os.path.join(folder, fname), "wb") as f:
                    f.write(img_data)
                count += 1
            except Exception as e:
                print(f"Error downloading {url}: {e}")
        if count >= IMAGES_PER_CATEGORY:
            break

    print(f"Category '{cat_name}' downloaded: {count} images.")

def main():
    for cat, query in CATEGORIES.items():
        print(f"Downloading category: {cat}")
        download_category(cat, query)

if __name__ == "__main__":
    main()
