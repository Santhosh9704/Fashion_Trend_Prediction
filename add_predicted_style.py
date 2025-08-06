import pandas as pd

# Load your caption_sentiment.csv
df = pd.read_csv("caption_sentiment.csv")

# Define simple keyword-based rules for style prediction
def predict_style(caption):
    caption = caption.lower()

    # Retro keywords
    if any(keyword in caption for keyword in ["retro", "vintage", "90s", "throwback"]):
        return "retro"
    
    # Streetwear keywords
    elif any(keyword in caption for keyword in ["street", "hoodie", "sneakers", "urban", "swag"]):
        return "streetwear"
    
    # Monochrome keywords
    elif any(keyword in caption for keyword in ["monochrome", "black and white", "minimal", "gray", "grey", "neutral"]):
        return "monochrome"
    
    # Fallback
    else:
        return "unknown"

# Apply to all captions
df["predicted_style"] = df["caption"].apply(predict_style)

# Save updated CSV
df.to_csv("caption_sentiment_with_style.csv", index=False)

print("✅ Styles predicted and saved to caption_sentiment_with_style.csv")
