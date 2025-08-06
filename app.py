import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import cv2
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="AI Fashion Trend Intelligence",
    layout="wide",
    page_icon="👗"
)

# Load model
model = tf.keras.models.load_model("style_classifier_model.h5")
class_names = ["retro", "streetwear", "monochrome"]

# Load sentiment data
sentiment_df = pd.read_csv("caption_sentiment.csv") if os.path.exists("caption_sentiment.csv") else pd.DataFrame()

# Tabs for navigation
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
        .css-18e3th9 {
            padding-top: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("👗 AI Fashion Trend Intelligence")
tabs = st.tabs(["Style Prediction", "Social Sentiment", "Style Browser"])

# ---------- TAB 1: STYLE PREDICTION ---------- #
with tabs[0]:
    st.subheader("Upload an Image to Predict Style")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)

        prediction = model.predict(img_input)
        predicted_class = class_names[np.argmax(prediction)]

        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.success(f"🧠 Predicted Style: **{predicted_class.upper()}**")

# ---------- TAB 2: SENTIMENT DASHBOARD ---------- #
with tabs[1]:
    st.subheader("Social Sentiment Analysis")
    if not sentiment_df.empty:
        sentiment_counts = sentiment_df["sentiment"].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Top Performing Post")
        top_post = sentiment_df.loc[sentiment_df["likes"].idxmax()]
        st.write(f"**User:** {top_post['username']} | **Platform:** {top_post['platform']} | **Likes:** {top_post['likes']}")
        st.info(top_post['caption'])
        st.success(f"Sentiment: {top_post['sentiment'].capitalize()}")
    else:
        st.warning("No sentiment data found. Please run sentiment_analysis.py")

# ---------- TAB 3: STYLE BROWSER ---------- #
with tabs[2]:
    st.subheader("Explore Example Styles")
    col1, col2, col3 = st.columns(3)
    style_folder = "fashion_styles"

    for i, (label, col) in enumerate(zip(class_names, [col1, col2, col3])):
        subfolder = os.path.join(style_folder, label)
        images = os.listdir(subfolder)[:3] if os.path.exists(subfolder) else []
        with col:
            st.markdown(f"#### {label.title()}")
            for img_file in images:
                st.image(os.path.join(subfolder, img_file), use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ for Fashion dheepsam 2025")
