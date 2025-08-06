import pandas as pd
import re

# Create sample social post data
data = {
    "platform": ["Instagram", "Twitter", "Pinterest"],
    "username": ["fashionista_101", "runwayqueen", "fab_finds"],
    "caption": [
        "Loving this oversized blazer 🔥 #OOTD #StreetStyle",
        "Back to 90s with #Retro looks! ✨🖤",
        "Lace is back in trend! Loving these feminine vibes 💃 #LaceTrend"
    ],
    "likes": [1250, 430, 212],
    "hashtags": ["#OOTD,#StreetStyle", "#Retro", "#LaceTrend"]
}

df = pd.DataFrame(data)

# Clean captions
def clean_caption(text):
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower().strip()

df["cleaned_caption"] = df["caption"].apply(clean_caption)

# Save to file
df.to_csv("cleaned_social_posts.csv", index=False)
print("✅ File created: cleaned_social_posts.csv")
