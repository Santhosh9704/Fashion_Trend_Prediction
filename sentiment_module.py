import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Configuration & Helpers ---
st.set_page_config(layout="wide", page_title="Fashion Sentiment")

def load_data(filepath="caption_sentiment.csv"):
    """Loads sentiment data, handling file not found and empty file cases."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if not df.empty:
            return df
    return None

def get_sentiment_emoji(sentiment):
    """Returns an emoji for a given sentiment string."""
    emoji_map = {
        "positive": "✅",
        "negative": "❌",
        "neutral": "➖"
    }
    return emoji_map.get(sentiment.lower(), "❓")

# --- UI Components ---

def display_header():
    """Displays the main header of the dashboard."""
    st.title("🧠 Fashion Trend Sentiment Dashboard")
    st.write("Analyze fashion post sentiment across social platforms.")
    st.markdown("---")

def display_kpis(df):
    """Displays Key Performance Indicators."""
    total_posts = len(df)
    total_likes = df["likes"].sum()
    avg_likes = df["likes"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts Analyzed", f"{total_posts:,}")
    col2.metric("Total Likes", f"{int(total_likes):,}")
    col3.metric("Average Likes per Post", f"{avg_likes:,.2f}")
    st.markdown("---")

def display_sentiment_pie(df):
    """Displays a pie chart for sentiment distribution."""
    st.subheader("Sentiment Breakdown")
    sentiment_counts = df["sentiment"].value_counts()
    fig = px.pie(
        sentiment_counts,
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Overall Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map={
            "Positive": "lightgreen",
            "Negative": "lightcoral",
            "Neutral": "lightskyblue"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

def display_top_post(df):
    """Displays the most liked post."""
    st.subheader("🔥 Top Performing Post")
    if "likes" in df.columns and not df["likes"].empty:
        top_post = df.loc[df["likes"].idxmax()]

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Likes", f"{top_post['likes']:,}")
            st.markdown(f"**Platform:** {top_post['platform']}")
            st.markdown(f"**User:** {top_post['username']}")

        with col2:
            st.markdown(f"**Caption:**")
            st.info(f"_{top_post['caption']}_")
            sentiment_emoji = get_sentiment_emoji(top_post['sentiment'])
            st.success(f"**Sentiment:** {top_post['sentiment'].capitalize()} {sentiment_emoji}")
    else:
        st.warning("No 'likes' data available to determine the top post.")

def display_posts_table(df):
    """Displays a filterable table of all posts."""
    st.subheader("Explore All Posts")

    # Add a filter for platform
    platforms = ["All"] + df["platform"].unique().tolist()
    selected_platform = st.selectbox("Filter by Platform", platforms)

    if selected_platform == "All":
        filtered_df = df
    else:
        filtered_df = df[df["platform"] == selected_platform]

    st.dataframe(
        filtered_df[["platform", "username", "likes", "cleaned_caption", "sentiment"]],
        use_container_width=True
    )

# --- Main App Logic ---
def main():
    """Main function to run the Streamlit app."""
    df = load_data()

    display_header()

    if df is not None:
        display_kpis(df)
        display_sentiment_pie(df)
        display_top_post(df)
        st.markdown("---")
        display_posts_table(df)
    else:
        st.error("❌ **Data not found!** Please run `sentiment_analysis.py` to generate `caption_sentiment.csv`.")

if __name__ == "__main__":
    main()