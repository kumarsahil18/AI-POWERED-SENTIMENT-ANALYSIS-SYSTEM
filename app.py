import streamlit as st
import pickle
import math
import re
from collections import Counter

st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

# Load trained model
with open("model/sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

word_counts = {
    label: Counter(words)
    for label, words in model["word_counts"].items()
}

class_counts = Counter(model["class_counts"])
classes = model["classes"]
vocabulary_size = model["vocabulary_size"]


def clean_text(text):
    text = text.lower()
    return re.findall(r"[a-z]+", text)


def predict(text):
    words = clean_text(text)
    scores = {}

    total_documents = sum(class_counts.values())

    for label in classes:
        score = math.log(
            class_counts[label] / total_documents
        )

        total_words = sum(word_counts[label].values())

        for word in words:
            probability = (
                word_counts[label][word] + 1
            ) / (
                total_words + vocabulary_size
            )

            score += math.log(probability)

        scores[label] = score

    return max(scores, key=scores.get)


# -------------------------------
# Session History
# -------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------------
# UI
# -------------------------------

st.title("🤖 AI-Powered Sentiment Analysis System")

st.write(
    "Analyze the sentiment of your text using an AI-based "
    "Machine Learning model."
)

st.divider()

text = st.text_area(
    "✍️ Enter your text",
    placeholder="Example: I really enjoyed this product..."
)

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if text.strip():

        result = predict(text)

        # Save history
        st.session_state.history.append(
            {
                "Text": text,
                "Sentiment": result.capitalize()
            }
        )

        st.subheader("Result")

        if result == "positive":
            st.success("😊 Positive Sentiment")

        elif result == "negative":
            st.error("😞 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")

    else:
        st.warning("⚠️ Please enter some text first.")


# -------------------------------
# History
# -------------------------------

if st.session_state.history:

    st.divider()

    st.subheader("📊 Sentiment History")

    for i, item in enumerate(
        reversed(st.session_state.history), 1
    ):
        st.write(
            f"**{i}.** {item['Text']} → "
            f"**{item['Sentiment']}**"
        )


st.divider()

st.caption(
    "AI-Powered Sentiment Analysis System | "
    "Python + Machine Learning"
)