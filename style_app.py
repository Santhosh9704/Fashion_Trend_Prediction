import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import cv2
import os
import base64
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="AI Fashion Trend Intelligence",
    layout="wide",
    page_icon="👗"
)

# -------------------- Background Styling --------------------
def add_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <style>
                .stApp {{
                    background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url('data:image/png;base64,{encoded}');
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                    color: white;
                    zoom: 1.1;
                }}
                .main-title {{
                    font-size: 2.8rem;
                    font-weight: 900;
                    text-align: left;
                    color: #ffffff;
                    font-family: 'Segoe UI', sans-serif;
                    padding: 20px 0 30px 10px;
                }}
                .stTabs [data-baseweb="tab"] {{
                    font-size: 22px !important;
                    font-weight: 700 !important;
                    padding: 14px 28px !important;
                    color: #f0f0f0 !important;
                    font-family: 'Segoe UI', sans-serif;
                    border-bottom: 3px solid transparent;
                }}
                .stTabs [aria-selected="true"] {{
                    border-bottom: 4px solid #ff4b4b !important;
                    color: white !important;
                }}
                h2, h3, h4, h5, h6 {{
                    color: #ffffff !important;
                }}
                .stButton>button, .stFileUploader, .stDataFrame, .stImage>img {{
                    background-color: rgba(255, 255, 255, 0.92);
                    color: black !important;
                    border-radius: 10px;
                    padding: 10px;
                }}
            </style>
        """, unsafe_allow_html=True)

add_background("online-shopping-concept.jpg")

# -------------------- Load Model & Data --------------------
model = tf.keras.models.load_model("style_classifier_model.keras")
class_names = ["retro", "streetwear", "monochrome"]
sentiment_df = pd.read_csv("caption_sentiment.csv") if os.path.exists("caption_sentiment.csv") else pd.DataFrame()

# -------------------- Title --------------------
st.markdown('<div class="main-title">👗 AI Fashion Trend Intelligence</div>', unsafe_allow_html=True)

# -------------------- Tabs --------------------
tabs = st.tabs([
    "🏠 Home", "🧠 Style Prediction", "💬 Social Sentiment", "🖼️ Style Browser", "📊 EDA & Evaluation", "ℹ️ About"
])

# -------------------- HOME TAB --------------------
with tabs[0]:
    st.subheader("👋 Welcome to AI Fashion Intelligence")
    st.markdown("""
        Discover trending styles and analyze fashion sentiment using AI.

        ### 🔍 Features:
        - Upload images to detect fashion style
        - Analyze public sentiment on Instagram, Twitter & Pinterest
        - View visually categorized styles
        - See AI training performance & visual evaluation
    """)

# -------------------- STYLE PREDICTION --------------------
with tabs[1]:
    st.subheader("📤 Upload an Image to Predict Style")
    uploaded_file = st.file_uploader("Choose a fashion image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)

        prediction = model.predict(img_input)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.image(img, caption="📸 Uploaded Image", width=400)
        st.success(f"🧠 Predicted Style: **{predicted_class.upper()}**")
        st.progress(confidence)
        st.caption(f"🔍 Confidence: {confidence:.2%}")

# -------------------- SOCIAL SENTIMENT --------------------
with tabs[2]:
    st.subheader("💬 Social Sentiment Dashboard")
    if not sentiment_df.empty:
        sentiment_counts = sentiment_df["sentiment"].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True, key="sentiment_pie")

        st.markdown("### 🔥 Top Performing Post")
        top_post = sentiment_df.loc[sentiment_df["likes"].idxmax()]
        st.markdown(f"**User:** {top_post['username']}")
        st.markdown(f"**Platform:** {top_post['platform']}")
        st.markdown(f"**Likes:** {top_post['likes']}")
        st.info(top_post['caption'])
        st.success(f"Sentiment: {top_post['sentiment'].capitalize()}")
    else:
        st.warning("⚠️ No sentiment data found. Please run `sentiment_analysis.py`.")

# -------------------- STYLE BROWSER --------------------
with tabs[3]:
    st.subheader("🖼️ Browse Fashion Styles by Category")
    style_folder = "fashion_styles"
    retro_imgs = sorted(os.listdir(os.path.join(style_folder, "retro"))) if os.path.exists(os.path.join(style_folder, "retro")) else []
    street_imgs = sorted(os.listdir(os.path.join(style_folder, "streetwear"))) if os.path.exists(os.path.join(style_folder, "streetwear")) else []
    mono_imgs = sorted(os.listdir(os.path.join(style_folder, "monochrome"))) if os.path.exists(os.path.join(style_folder, "monochrome")) else []

    total = min(len(retro_imgs), len(street_imgs), len(mono_imgs))
    for i in range(total):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 👗 Retro")
            st.image(os.path.join(style_folder, "retro", retro_imgs[i]), width=250)
        with col2:
            st.markdown("##### 👕 Streetwear")
            st.image(os.path.join(style_folder, "streetwear", street_imgs[i]), width=250)
        with col3:
            st.markdown("##### 🖤 Monochrome")
            st.image(os.path.join(style_folder, "monochrome", mono_imgs[i]), width=250)

# -------------------- EDA & EVALUATION --------------------
with tabs[4]:
    st.subheader("📊 Exploratory Data Analysis & Evaluation")

    if not sentiment_df.empty:
        st.markdown("### 📈 Dataset Preview")
        st.dataframe(sentiment_df.head())

        st.markdown("### 🔍 Sentiment Distribution")
        st.bar_chart(sentiment_df["sentiment"].value_counts())

        st.markdown("### 📌 Likes Distribution")
        st.line_chart(sentiment_df["likes"].sort_values().reset_index(drop=True))

        # ✅ HEATMAP based on sentiment vs style
        if 'predicted_style' in sentiment_df.columns:
            st.markdown("### 🌡️ Sentiment Heatmap by Style")
            heat_data = pd.crosstab(sentiment_df['predicted_style'], sentiment_df['sentiment'])
            fig_heatmap = px.imshow(heat_data, text_auto=True, color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("⚠️ 'predicted_style' column not found in sentiment data.")

    else:
        st.warning("⚠️ Please run `sentiment_analysis.py` to generate sentiment data.")

    # Static evaluation visuals
    if os.path.exists("confusion_matrix.png"):
        st.markdown("### 🧮 Confusion Matrix")
        st.image("confusion_matrix.png")

    if os.path.exists("roc_auc_curve.png"):
        st.markdown("### 📈 ROC-AUC Curve")
        st.image("roc_auc_curve.png")

    if os.path.exists("accuracy_plot.png"):
        st.markdown("### 🔁 Model Training Accuracy")
        st.image("accuracy_plot.png")

# -------------------- ABOUT TAB --------------------
with tabs[5]:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    **AI Fashion Trend Intelligence** uses AI to identify fashion styles and analyze sentiment from social media platforms.

    ### 💡 What It Does:
    - Predicts fashion styles (Retro, Streetwear, Monochrome)
    - Detects top-performing posts
    - Visualizes training metrics and user sentiments

    ### 🛠️ Technologies:
    - TensorFlow, Keras, OpenCV
    - Streamlit, Pandas, Plotly
    """)

# -------------------- Footer --------------------
st.markdown("---")
st.caption("👠 Made with ❤️ for Fashion 2025 by chubby")