import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("caption_sentiment.csv")

# Split hashtags by comma and flatten list
hashtags = []
for tag_list in df["hashtags"].dropna():
    tags = tag_list.lower().split(",")
    hashtags.extend([tag.strip() for tag in tags if tag.strip() != ""])

# Count frequency
top_tags = Counter(hashtags).most_common(10)

# Display
print("📊 Top Hashtags:")
for tag, count in top_tags:
    print(f"{tag}: {count} times")

# Plot
tags, counts = zip(*top_tags)
plt.figure(figsize=(8, 4))
plt.barh(tags, counts, color="skyblue")
plt.xlabel("Frequency")
plt.title("Top 10 Fashion Hashtags")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_hashtags.png")
plt.show()
