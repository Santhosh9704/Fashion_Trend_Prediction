import os
import requests
from bs4 import BeautifulSoup

url = "https://www.vogue.com/fashion-shows/spring-2024-ready-to-wear"
folder = "runway_images"
os.makedirs(folder, exist_ok=True)

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

images = soup.find_all("img")

for i, img in enumerate(images[:30]):
    try:
        img_url = img.get('src')
        if not img_url:
            continue

        # Construct absolute URL if it's a relative path
        if not img_url.startswith(('http://', 'https://')):
            img_url = requests.compat.urljoin("https://www.vogue.com", img_url)

        img_data = requests.get(img_url).content
        with open(f"{folder}/image_{i}.jpg", 'wb') as f:
            f.write(img_data)
        print(f"Downloaded image_{i}.jpg")
    except Exception as e:
        print(f"Failed to download image {i}: {e}")
