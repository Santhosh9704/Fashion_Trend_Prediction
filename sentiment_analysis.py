import pandas as pd
from textblob import TextBlob
import os

# Check current directory
print("📁 Current working directory:", os.getcwd())

# Load the CSV file
try:
    df = pd.read_csv("cleaned_social_posts.csv")
    print(f"✅ Loaded {len(df)} rows from cleaned_social_posts.csv")
except FileNotFoundError:
    print("❌ ERROR: cleaned_social_posts.csv not found!")
    exit()

# Function to get sentiment label
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply to each caption
df["sentiment"] = df["cleaned_caption"].apply(get_sentiment)

# Save to new file
df.to_csv(r"C:\Users\santh\Downloads\FashionTrendProject\caption_sentiment.csv", index=False)
print("✅ Sentiment file saved as caption_sentiment.csv")
