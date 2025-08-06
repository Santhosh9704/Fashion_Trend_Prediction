import pandas as pd

# Load sentiment data
sentiment_df = pd.read_csv("caption_sentiment.csv")

# Filter for positive posts (may be 0)
positive_posts = sentiment_df[sentiment_df["sentiment"] == "positive"]

print("📈 Recommendations:\n")

# ✅ Keyword Suggestions
if not positive_posts.empty:
    most_common_words = (
        positive_posts["cleaned_caption"]
        .str.split()
        .explode()
        .value_counts()
        .head(5)
    )
    top_platform = positive_posts["platform"].value_counts().idxmax()

    print("1️⃣ Focus on content with these keywords:", ", ".join(most_common_words.index))
    print(f"2️⃣ Publish more on {top_platform} – your most positively engaged platform.")
else:
    print("⚠️ No positive sentiment data found. Try collecting more posts or checking the sentiment analysis step.")
    print("1️⃣ Focus on increasing audience engagement through polls, questions, or influencer content.")
    print("2️⃣ Try re-running `sentiment_analysis.py` after expanding your dataset.")

# Always show visual references
print("3️⃣ Consider promoting **Retro** and **Streetwear** — these show higher classification accuracy.")
print("4️⃣ Visual hashtag trend image → 📷 top_hashtags.png")
print("5️⃣ See model training curve → 📈 accuracy_plot.png")
