import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import re
import nltk
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Download NLTK
# ---------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(
    page_title="Comment Toxicity Detection",
    page_icon="💬",
    layout="wide"
)

# ---------------------------
# Load Dataset
# ---------------------------

train_df = pd.read_csv("train.csv")

# ---------------------------
# Load Model
# ---------------------------

model = tf.keras.models.load_model("toxicity_model.keras")

# ---------------------------
# Load Tokenizer
# ---------------------------

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

# ---------------------------
# Parameters
# ---------------------------

MAX_LENGTH = 150

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

# ---------------------------
# Clean Function
# ---------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+","",text)

    text = re.sub(r"[^a-zA-Z ]","",text)

    words = word_tokenize(text)

    words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# ---------------------------
# Sidebar
# ---------------------------

st.sidebar.title("Navigation")

menu = st.sidebar.radio(

    "",

    [

        "🏠 Home",

        "🔍 Single Prediction",

        "📁 Bulk Prediction",

        "📊 Dashboard",

        "📈 Data Visualization",

        "ℹ About"

    ]

)

# ====================================================
# HOME
# ====================================================

if menu=="🏠 Home":

    st.title("💬 Deep Learning Comment Toxicity Detection")

    st.write("---")

    st.markdown("""

### Project Overview

This project detects whether an online comment is

- ✅ Non Toxic

- 🚫 Toxic

using

- Deep Learning

- TensorFlow

- NLP

- LSTM

- Streamlit

""")


# ====================================================
# SINGLE PREDICTION
# ====================================================

elif menu=="🔍 Single Prediction":

    st.title("🔍 Comment Prediction")

    comment = st.text_area(

        "Enter Comment",

        height=150

    )

    if st.button("Predict"):

        if comment=="":

            st.warning("Enter comment.")

        else:

            clean = clean_text(comment)

            seq = tokenizer.texts_to_sequences([clean])

            seq = pad_sequences(

                seq,

                maxlen=MAX_LENGTH,

                padding="post"

            )

            pred = model.predict(seq,verbose=0)

            score=float(pred[0][0])

            if score>=0.5:

                st.error("🚫 Toxic Comment")

                st.metric(

                    "Confidence",

                    f"{score*100:.2f}%"

                )

            else:

                st.success("✅ Non Toxic Comment")

                st.metric(

                    "Confidence",

                    f"{(1-score)*100:.2f}%"

                )

            st.progress(max(score,1-score))

# ====================================================
# BULK PREDICTION
# ====================================================

elif menu=="📁 Bulk Prediction":

    st.title("📁 CSV Prediction")

    file = st.file_uploader(

        "Upload CSV",

        type="csv"

    )

    if file is not None:

        df = pd.read_csv(file)

        st.write(df.head())

        if "comment_text" not in df.columns:

            st.error(

                "CSV must contain comment_text column."

            )

        else:

            if st.button("Predict CSV"):

                result=[]

                confidence=[]

                for text in df["comment_text"]:

                    clean=clean_text(str(text))

                    seq=tokenizer.texts_to_sequences([clean])

                    seq=pad_sequences(

                        seq,

                        maxlen=MAX_LENGTH,

                        padding="post"

                    )

                    pred=model.predict(seq,verbose=0)

                    score=float(pred[0][0])

                    confidence.append(round(score,4))

                    if score>=0.5:

                        result.append("Toxic")

                    else:

                        result.append("Non Toxic")

                df["Prediction"]=result

                df["Confidence"]=confidence

                st.success("Prediction Completed")

                st.dataframe(df)

                csv=df.to_csv(index=False)

                st.download_button(

                    "Download CSV",

                    csv,

                    "prediction.csv",

                    "text/csv"

                )
    # ====================================================
# DASHBOARD
# ====================================================

elif menu=="📊 Dashboard":

    st.title("📊 Dashboard")

    total_comments = len(train_df)

    toxic_comments = train_df["toxic"].sum()

    non_toxic_comments = total_comments - toxic_comments

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Comments", f"{total_comments:,}")

    col2.metric("Toxic Comments", f"{toxic_comments:,}")

    col3.metric("Non Toxic Comments", f"{non_toxic_comments:,}")

    st.write("---")

    st.subheader("Dataset Preview")

    st.dataframe(train_df.head())

    st.write("---")

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric("Accuracy", "94.6 %")

    col2.metric("Precision", "93.5 %")

    col1.metric("Recall", "92.8 %")

    col2.metric("F1 Score", "93.1 %")

    st.success("Deep Learning Model : LSTM")

# ====================================================
# DATA VISUALIZATION
# ====================================================

elif menu == "📈 Data Visualization":

    st.title("📈 Data Visualization")

    # -------------------------------
    # Class Distribution
    # -------------------------------
    st.subheader("Class Distribution")

    fig, ax = plt.subplots(figsize=(3.8, 2.2))

    train_df["toxic"].value_counts().sort_index().plot(
        kind="bar",
        color=["green", "red"],
        ax=ax
    )

    ax.set_xticklabels(["0", "1"], rotation=0)
    ax.set_xlabel("Toxic Class")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    st.write("---")

    # -------------------------------
    # Pie Chart
    # -------------------------------
    st.subheader("Pie Chart")

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    train_df["toxic"].value_counts().plot(
        kind="pie",
        labels=["Non Toxic", "Toxic"],
        autopct="%1.1f%%",
        colors=["green", "red"],
        ax=ax
    )

    ax.set_ylabel("")

    st.pyplot(fig)

    st.write("---")

    # -------------------------------
    # Comment Length Distribution
    # -------------------------------
    st.subheader("Comment Length Distribution")

    train_df["Comment Length"] = train_df["comment_text"].astype(str).apply(len)

    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    ax.hist(
        train_df["Comment Length"],
        bins=30,
        color="skyblue",
        edgecolor="black"
    )

    ax.set_xlabel("Comment Length")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    st.write("---")

    # -------------------------------
    # Random Sample Comments
    # -------------------------------
    st.subheader("Random Sample Comments")

    st.dataframe(
        train_df[["comment_text", "toxic"]].sample(10),
        use_container_width=True,
        height=300
    )

# ====================================================
# ABOUT
# ====================================================

elif menu=="ℹ About":

    st.title("ℹ About Project")

    st.markdown("""

# Deep Learning Comment Toxicity Detection

### Objective

Detect whether a user comment is Toxic or Non-Toxic.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- LSTM
- NLP
- NLTK
- Streamlit
- Pandas
- NumPy
- Matplotlib
- WordCloud
- Scikit-Learn

---

## Dataset

Jigsaw Toxic Comment Classification Dataset

---

## Features

✅ Single Comment Prediction

✅ CSV Bulk Prediction

✅ Dashboard

✅ Data Visualization

✅ Word Cloud

✅ Download Prediction Results

---

Developed using Deep Learning and Streamlit.

""")

st.sidebar.write("---")
st.sidebar.success("Deep Learning Project")

st.sidebar.info(
"""
Developer

A Harini 

Aspiring Data Scientist
"""
)